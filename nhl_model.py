#!/usr/bin/env python3
"""
NHL Game-Line Sharp Model
-------------------------
Flow (each run):
  1. NHL schedule for today (ET). Skip games already posted or already started.
  2. Check DailyFaceoff starting goalies (+ goalies_override.json).
     -> If no game has BOTH goalies confirmed: exit. No stats pulled, no Odds API credits used.
  3. For newly confirmed games: pull MoneyPuck team xG + goalie GSAx (current season blended w/ prior).
  4. Monte Carlo (regulation Poisson + empty-net + 3v3 OT + shootout, sportsbook settlement rules).
  5. Pull Odds API (Pinnacle + your books), power-devig Pinnacle, compare every book price to fair.
  6. Post to Discord, log plays to CSV, mark games posted in nhl_state.json.

Env vars: ODDS_API_KEY, DISCORD_WEBHOOK_URL, FORCE=1 (optional: ignore goalie gate, use projected)
"""
import os, re, csv, io, json, math, unicodedata, datetime as dt
from zoneinfo import ZoneInfo
import requests
import numpy as np

# ============================ CONFIG ============================
ODDS_API_KEY    = os.getenv("ODDS_API_KEY", "")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "")
FORCE           = os.getenv("FORCE", "0") == "1"

SHARP_BOOK = "pinnacle"
# Match this list to your Game-Line-Pinny-Devig KS books
BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us",
         "fanatics", "espnbet", "betrivers", "novig"]
MARKETS = "h2h,spreads,totals"

CONFIRM_LEVELS = {"confirmed"}      # add "likely" to run earlier on strong reports
N_SIMS         = 20000
MODEL_WEIGHT   = 0.30   # final prob = 70% Pinnacle fair + 30% model
MIN_EDGE       = 0.015  # blended EV needed for a tiered play
STALE_MIN      = 15     # ignore book prices older than this vs Pinnacle (minutes)
BANKROLL       = 2000
KELLY_FRAC     = 0.25
MAX_STAKE_PCT  = 0.03

# Model knobs (tuned by backtest: 2024-25 + 2025-26, 2,624 games)
HFA            = 1.035  # home scoring multiplier (away gets 1/HFA)
PRIOR_GAMES    = 10     # current-season GP needed to equal prior-season weight
PRIOR_REGRESS  = 0.33   # regress prior season 1/3 to league mean (offseason churn)
XG_WEIGHT      = 0.55   # offense = 55% xGF + 45% actual GF (finishing talent)
GOALIE_K       = 350    # GSAx shrinkage (xGA faced)
UNKNOWN_GOALIE = 1.03   # factor for goalie with no NHL data (call-up)
B2B_PENALTY    = 0.07   # back-to-back: -7% own scoring, +7% allowed
# Travel (0 = off until the backtest measures them; paste tuned values here)
B2B_TRAVEL_K   = 0.0    # extra B2B penalty per 1,000 miles flown since last game
TZ_K           = 0.0    # penalty per time zone crossed (last game <= 2 days ago)
DENSE_K        = 0.0    # penalty when tonight is the 3rd game in 4 nights
FATIGUE_CAP    = 0.25   # max total fatigue penalty
ENG_PROB       = 0.08   # P(empty-net goal | 1-goal lead after regulation)
OT_GOAL_PROB   = 0.672  # P(OT decided before shootout)

ET = ZoneInfo("America/New_York")
STATE_FILE, LOG_FILE, OVERRIDE_FILE = "nhl_state.json", "nhl_plays_log.csv", "goalies_override.json"
UA = {"User-Agent": "Mozilla/5.0 (nhl-model)"}

TEAMS = {  # Odds API / DailyFaceoff full names -> NHL abbrev
    "anaheim ducks": "ANA", "boston bruins": "BOS", "buffalo sabres": "BUF",
    "calgary flames": "CGY", "carolina hurricanes": "CAR", "chicago blackhawks": "CHI",
    "colorado avalanche": "COL", "columbus blue jackets": "CBJ", "dallas stars": "DAL",
    "detroit red wings": "DET", "edmonton oilers": "EDM", "florida panthers": "FLA",
    "los angeles kings": "LAK", "minnesota wild": "MIN", "montreal canadiens": "MTL",
    "nashville predators": "NSH", "new jersey devils": "NJD", "new york islanders": "NYI",
    "new york rangers": "NYR", "ottawa senators": "OTT", "philadelphia flyers": "PHI",
    "pittsburgh penguins": "PIT", "san jose sharks": "SJS", "seattle kraken": "SEA",
    "st louis blues": "STL", "tampa bay lightning": "TBL", "toronto maple leafs": "TOR",
    "utah mammoth": "UTA", "utah hockey club": "UTA", "vancouver canucks": "VAN",
    "vegas golden knights": "VGK", "washington capitals": "WSH", "winnipeg jets": "WPG",
}
MP_FIX = {"N.J": "NJD", "T.B": "TBL", "L.A": "LAK", "S.J": "SJS", "ARI": "UTA"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower().replace(".", "")).strip()


def team_abbr(name):
    n = norm(name)
    if n in TEAMS:
        return TEAMS[n]
    for full, ab in TEAMS.items():  # fallback: nickname match ("Maple Leafs")
        if full.split(" ", 1)[-1] in n or n.endswith(full.split()[-1]):
            return ab
    return None


def get(url, **kw):
    r = requests.get(url, headers=UA, timeout=25, **kw)
    r.raise_for_status()
    return r


# ============================ ODDS MATH ============================
def am_to_dec(a):  return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)
def dec_to_am(d):  return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))
def p_to_am(p):    return dec_to_am(1 / p)


def power_devig(implied):
    """Find k so sum(p_i^k)=1. Handles favorite-longshot bias better than multiplicative."""
    lo, hi = 0.01, 20.0
    for _ in range(100):
        k = (lo + hi) / 2
        if sum(p ** k for p in implied) > 1: lo = k
        else: hi = k
    out = [p ** k for p in implied]
    s = sum(out)
    return [p / s for p in out]


def kelly_stake(p, dec):
    b = dec - 1
    f = (p * b - (1 - p)) / b
    return max(0.0, min(f * KELLY_FRAC, MAX_STAKE_PCT)) * BANKROLL


# ============================ GOALIES ============================
def fetch_dailyfaceoff(date_str):
    """Returns {team_abbr: {"goalie": name, "status": "confirmed|likely|unconfirmed"}}"""
    out = {}
    try:
        html = get(f"https://www.dailyfaceoff.com/starting-goalies/{date_str}").text
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
        data = json.loads(m.group(1))
    except Exception as e:
        print(f"[goalies] DailyFaceoff fetch/parse failed: {e}")
        return out

    def pick(d, side, *need, avoid=()):
        for k, v in d.items():
            kl = k.lower()
            if kl.startswith(side) and all(n in kl for n in need) and not any(a in kl for a in avoid):
                return v
        return None

    def walk(o):
        if isinstance(o, dict):
            if any(k.lower().startswith("home") and "goalie" in k.lower() for k in o):
                for side in ("home", "away"):
                    team = pick(o, side, "team", "name", avoid=("goalie",)) or pick(o, side, "team", avoid=("goalie", "id", "logo"))
                    goalie = pick(o, side, "goalie", "name")
                    status = pick(o, side, "strength", "name") or pick(o, side, "status") or ""
                    ab = team_abbr(team) if isinstance(team, str) else None
                    if ab and isinstance(goalie, str):
                        out[ab] = {"goalie": goalie, "status": str(status).lower()}
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)

    walk(data)
    print(f"[goalies] DailyFaceoff parsed {len(out)} teams")
    return out


def load_overrides():
    """goalies_override.json: {"BOS": "Jeremy Swayman"} -> treated as confirmed today."""
    try:
        with open(OVERRIDE_FILE) as f:
            return {k.upper(): {"goalie": v, "status": "confirmed"} for k, v in json.load(f).items() if v}
    except FileNotFoundError:
        return {}


# ============================ STATS (MoneyPuck) ============================
def mp_csv(season, kind):
    try:
        txt = get(f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/{kind}.csv").text
        return [r for r in csv.DictReader(io.StringIO(txt)) if r.get("situation") == "all"]
    except Exception as e:
        print(f"[stats] MoneyPuck {season} {kind} unavailable: {e}")
        return []


def f(r, k):
    try: return float(r.get(k) or 0)
    except ValueError: return 0.0


def build_team_ratings(season):
    cur, prev = mp_csv(season, "teams"), mp_csv(season - 1, "teams")

    def per_game(rows):
        out = {}
        for r in rows:
            gp = f(r, "games_played")
            if gp <= 0: continue
            ab = MP_FIX.get(r["team"], r["team"])
            out[ab] = {"gp": gp,
                       "off": XG_WEIGHT * f(r, "xGoalsFor") / gp + (1 - XG_WEIGHT) * f(r, "goalsFor") / gp,
                       "def": f(r, "xGoalsAgainst") / gp,
                       "gf": f(r, "goalsFor") / gp, "xgf": f(r, "xGoalsFor") / gp}
        return out

    c, p = per_game(cur), per_game(prev)
    base = c if len(c) >= 30 else p
    lg_off = np.mean([t["off"] for t in base.values()])
    lg_def = np.mean([t["def"] for t in base.values()])
    lg_goals = np.mean([t["gf"] for t in base.values()])

    ratings = {}
    for ab in set(c) | set(p) | set(TEAMS.values()):
        po = p.get(ab, {}).get("off", lg_off) * (1 - PRIOR_REGRESS) + lg_off * PRIOR_REGRESS
        pd_ = p.get(ab, {}).get("def", lg_def) * (1 - PRIOR_REGRESS) + lg_def * PRIOR_REGRESS
        gp = c.get(ab, {}).get("gp", 0)
        w = gp / (gp + PRIOR_GAMES)
        off = w * c[ab]["off"] + (1 - w) * po if gp else po
        dfn = w * c[ab]["def"] + (1 - w) * pd_ if gp else pd_
        ratings[ab] = {"off": off / lg_off, "def": dfn / lg_def, "gp": gp}
    return ratings, lg_goals


def build_goalie_ratings(season):
    agg = {}
    for rows, wt in ((mp_csv(season, "goalies"), 1.0), (mp_csv(season - 1, "goalies"), 0.6)):
        for r in rows:
            key = norm(r.get("name", ""))
            a = agg.setdefault(key, {"xga": 0.0, "gsax": 0.0, "name": r.get("name")})
            a["xga"] += wt * f(r, "xGoals")
            a["gsax"] += wt * (f(r, "xGoals") - f(r, "goals"))
    for a in agg.values():
        rate = a["gsax"] / (a["xga"] + GOALIE_K)
        a["factor"] = float(np.clip(1 - rate, 0.85, 1.12))  # multiplies OPPONENT scoring
    return agg


def goalie_factor(name, goalies):
    n = norm(name)
    if n in goalies: return goalies[n]["factor"], True
    last = n.split()[-1] if n else ""
    hits = [g for k, g in goalies.items() if k.split() and k.split()[-1] == last]
    if len(hits) == 1: return hits[0]["factor"], True
    return UNKNOWN_GOALIE, False


# ============================ TRAVEL ============================
# abbrev: (lat, lon, tz offset vs Eastern)
ARENAS = {
    "ANA": (33.808, -117.877, -3), "BOS": (42.366, -71.062, 0), "BUF": (42.875, -78.876, 0),
    "CGY": (51.037, -114.052, -2), "CAR": (35.803, -78.722, 0), "CHI": (41.881, -87.674, -1),
    "COL": (39.749, -105.008, -2), "CBJ": (39.969, -83.006, 0), "DAL": (32.790, -96.810, -1),
    "DET": (42.341, -83.055, 0), "EDM": (53.547, -113.498, -2), "FLA": (26.158, -80.326, 0),
    "LAK": (34.043, -118.267, -3), "MIN": (44.945, -93.101, -1), "MTL": (45.496, -73.569, 0),
    "NSH": (36.159, -86.779, -1), "NJD": (40.734, -74.171, 0), "NYI": (40.712, -73.726, 0),
    "NYR": (40.751, -73.993, 0), "OTT": (45.297, -75.927, 0), "PHI": (39.901, -75.172, 0),
    "PIT": (40.439, -79.989, 0), "SJS": (37.333, -121.901, -3), "SEA": (47.622, -122.354, -3),
    "STL": (38.627, -90.203, -1), "TBL": (27.943, -82.452, 0), "TOR": (43.643, -79.379, 0),
    "UTA": (40.768, -111.901, -2), "VAN": (49.278, -123.109, -3), "VGK": (36.103, -115.178, -3),
    "WSH": (38.898, -77.021, 0), "WPG": (49.893, -97.144, -1),
}
NO_TRAVEL = dict(b2b=0, miles=0.0, tz=0, rest=9, dense=0)


def miles_between(a, b):
    if a not in ARENAS or b not in ARENAS or a == b: return 0.0
    la1, lo1, _ = ARENAS[a]; la2, lo2, _ = ARENAS[b]
    p1, p2, dl = math.radians(la1), math.radians(la2), math.radians(lo2 - lo1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return 3958.8 * math.acos(max(-1.0, min(1.0, c)))


def travel_features(past, loc, date_str):
    """past = sorted [(date, location_abbrev)] of the team's earlier games this season;
    loc = tonight's location (home team abbrev). Shared by live model and backtest."""
    if not past: return dict(NO_TRAVEL)
    d = dt.date.fromisoformat(date_str)
    last_d, last_loc = past[-1]
    rest = (d - dt.date.fromisoformat(last_d)).days
    tz = abs(ARENAS[last_loc][2] - ARENAS[loc][2]) if last_loc in ARENAS and loc in ARENAS else 0
    recent = sum(1 for x, _ in past[-3:] if 1 <= (d - dt.date.fromisoformat(x)).days <= 3)
    return dict(b2b=int(rest == 1), miles=miles_between(last_loc, loc), tz=tz, rest=rest, dense=int(recent >= 2))


def fatigue(t):
    f = (B2B_PENALTY + B2B_TRAVEL_K * t["miles"] / 1000) * t["b2b"] \
        + TZ_K * t["tz"] * (t["rest"] <= 2) + DENSE_K * t["dense"]
    return min(max(f, 0.0), FATIGUE_CAP)


def team_travel(team, loc, today, season):
    try:
        games = get(f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}{season + 1}").json().get("games", [])
    except Exception as e:
        print(f"[travel] {team} schedule failed: {e}"); return dict(NO_TRAVEL)
    past = sorted((g["gameDate"], g["homeTeam"]["abbrev"]) for g in games
                  if g.get("gameType") in (2, 3) and g["gameDate"] < today.isoformat())
    return travel_features(past, loc, today.isoformat())


def travel_note(team, t):
    bits = []
    if t["b2b"]: bits.append("B2B")
    if t["dense"]: bits.append("3-in-4")
    if t["miles"] >= 500 and t["rest"] <= 2: bits.append(f"{t['miles']:,.0f} mi")
    if t["tz"] and t["rest"] <= 2: bits.append(f"{t['tz']} TZ")
    return f"{team}: {', '.join(bits)}" if bits else ""


# ============================ SCHEDULE ============================
def nhl_games(date_str):
    data = get(f"https://api-web.nhle.com/v1/schedule/{date_str}").json()
    for day in data.get("gameWeek", []):
        if day.get("date") == date_str:
            return [g for g in day.get("games", []) if g.get("gameType") in (2, 3)]
    return []


# ============================ SIMULATION ============================
def simulate(lh, la, rng):
    hg, ag = rng.poisson(lh, N_SIMS), rng.poisson(la, N_SIMS)
    # empty-net: leader by 1 after regulation sometimes adds one
    eng = (np.abs(hg - ag) == 1) & (rng.random(N_SIMS) < ENG_PROB)
    hg = hg + (eng & (hg > ag))
    ag = ag + (eng & (ag > hg))
    tie = hg == ag
    ot_goal = rng.random(N_SIMS) < OT_GOAL_PROB
    ot_home = rng.random(N_SIMS) < lh / (lh + la)
    so_home = rng.random(N_SIMS) < 0.5
    home_wins_tie = np.where(ot_goal, ot_home, so_home)
    # sportsbook settlement: OT/SO winner gets +1 goal on final score
    hg = hg + (tie & home_wins_tie)
    ag = ag + (tie & ~home_wins_tie)
    return hg, ag


def model_prob(mkey, name, point, home, hg, ag):
    if mkey == "h2h":
        return float(np.mean(hg > ag)) if name == home else float(np.mean(ag > hg))
    if mkey == "spreads":
        margin = (hg - ag) if name == home else (ag - hg)
        win, lose = np.sum(margin + point > 0), np.sum(margin + point < 0)
    else:  # totals
        tot = hg + ag
        win = np.sum(tot > point) if name == "Over" else np.sum(tot < point)
        lose = np.sum(tot < point) if name == "Over" else np.sum(tot > point)
    return float(win / max(win + lose, 1))


# ============================ ODDS ============================
def fetch_odds():
    r = get("https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds",
            params={"apiKey": ODDS_API_KEY, "markets": MARKETS, "oddsFormat": "american",
                    "bookmakers": ",".join([SHARP_BOOK] + BOOKS)})
    print(f"[odds] credits remaining: {r.headers.get('x-requests-remaining')}")
    return r.json()


def parse_ts(s): return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def pinnacle_fair(event):
    pin = next((b for b in event["bookmakers"] if b["key"] == SHARP_BOOK), None)
    if not pin: return {}, None
    fair = {}
    for m in pin["markets"]:
        groups = {}
        for o in m["outcomes"]:
            groups.setdefault(abs(o["point"]) if "point" in o and m["key"] != "h2h" else None, []).append(o)
        for outs in groups.values():
            if len(outs) != 2: continue
            probs = power_devig([1 / am_to_dec(o["price"]) for o in outs])
            for o, p in zip(outs, probs):
                fair[(m["key"], o["name"], o.get("point"))] = p
    return fair, parse_ts(pin["last_update"])


# ============================ OUTPUT ============================
def post_discord(text):
    if not DISCORD_WEBHOOK:
        print(text); return
    for i in range(0, len(text), 1900):
        requests.post(DISCORD_WEBHOOK, json={"content": text[i:i + 1900]}, timeout=20)


def log_rows(rows):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new: w.writeheader()
        w.writerows(rows)


def label(mkey, name, point, abbr):
    if mkey == "h2h": return f"{abbr.get(name, name)} ML"
    if mkey == "spreads": return f"{abbr.get(name, name)} {point:+g}"
    return f"{name} {point:g}"


# ============================ MAIN ============================
def main():
    now = dt.datetime.now(dt.timezone.utc)
    today = now.astimezone(ET).date()
    ds = today.isoformat()

    state = {}
    if os.path.exists(STATE_FILE):
        state = json.load(open(STATE_FILE))
    state = {k: v for k, v in state.items() if v >= (today - dt.timedelta(days=3)).isoformat()}

    games = nhl_games(ds)
    pending = [g for g in games if str(g["id"]) not in state
               and parse_ts(g["startTimeUTC"]) > now + dt.timedelta(minutes=5)]
    if not pending:
        print("No pending games."); return

    goalies_today = fetch_dailyfaceoff(ds)
    goalies_today.update(load_overrides())

    ready, waiting = [], []
    for g in pending:
        h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        gh, ga = goalies_today.get(h), goalies_today.get(a)
        ok = gh and ga and gh["status"] in CONFIRM_LEVELS and ga["status"] in CONFIRM_LEVELS
        if ok or (FORCE and gh and ga): ready.append((g, gh, ga, ok))
        else: waiting.append(f"{a}@{h}")
    print(f"Ready: {len(ready)} | Waiting on goalies: {', '.join(waiting) or 'none'}")
    if not ready: return  # nothing pulled, no credits spent

    # ---- only now pull stats + odds ----
    season = today.year if today.month >= 9 else today.year - 1
    ratings, lg_goals = build_team_ratings(season)
    goalie_db = build_goalie_ratings(season)
    odds = fetch_odds()
    rng = np.random.default_rng()
    log = []

    for g, gh, ga, confirmed in ready:
        h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        ev = next((e for e in odds if team_abbr(e["home_team"]) == h and team_abbr(e["away_team"]) == a), None)
        if not ev:
            print(f"No odds for {a}@{h} yet"); continue
        abbr = {ev["home_team"]: h, ev["away_team"]: a}

        fh, kh = goalie_factor(gh["goalie"], goalie_db)
        fa, ka = goalie_factor(ga["goalie"], goalie_db)
        rh, ra = ratings[h], ratings[a]
        lh = lg_goals * rh["off"] * ra["def"] * HFA * fa
        la = lg_goals * ra["off"] * rh["def"] / HFA * fh
        th, ta = team_travel(h, h, today, season), team_travel(a, h, today, season)
        fh_, fa_ = fatigue(th), fatigue(ta)
        lh *= (1 - fh_) * (1 + fa_)
        la *= (1 - fa_) * (1 + fh_)
        hg, ag = simulate(lh, la, rng)
        p_home = float(np.mean(hg > ag))

        fair, pin_ts = pinnacle_fair(ev)
        if not fair:
            print(f"No Pinnacle for {a}@{h}"); continue

        plays = []
        for bm in ev["bookmakers"]:
            if bm["key"] == SHARP_BOOK: continue
            if abs((parse_ts(bm["last_update"]) - pin_ts).total_seconds()) > STALE_MIN * 60: continue
            for m in bm["markets"]:
                for o in m["outcomes"]:
                    key = (m["key"], o["name"], o.get("point"))
                    if key not in fair: continue
                    dec, pf = am_to_dec(o["price"]), fair[key]
                    pure_ev = pf * dec - 1
                    if pure_ev < 0: continue  # book doesn't match/beat true odds
                    pm = model_prob(m["key"], o["name"], o.get("point"), ev["home_team"], hg, ag)
                    pb = (1 - MODEL_WEIGHT) * pf + MODEL_WEIGHT * pm
                    bev = pb * dec - 1
                    agrees = pm > pf
                    tier = ("🟢 A" if bev >= .05 else "🟢 B" if bev >= .03 else "🟡 C" if bev >= MIN_EDGE else "⚪ match") if agrees else "⚪ price-only"
                    plays.append(dict(tier=tier, lbl=label(m["key"], o["name"], o.get("point"), abbr),
                                      mkey=m["key"], side=abbr.get(o["name"], o["name"]), point=o.get("point"),
                                      book=bm["title"], price=o["price"], fair=p_to_am(pf),
                                      pure=pure_ev, bev=bev, pm=pm, pf=pf,
                                      stake=kelly_stake(pb, dec) if agrees and bev >= MIN_EDGE else 0))
        plays.sort(key=lambda x: -x["bev"])
        # keep best book per outcome
        seen, best = set(), []
        for p in plays:
            if p["lbl"] in seen: continue
            seen.add(p["lbl"]); best.append(p)

        start = parse_ts(g["startTimeUTC"]).astimezone(ET).strftime("%-I:%M %p ET")
        gtag = "✅ confirmed" if confirmed else "⚠️ PROJECTED (forced)"
        hf = fair.get(("h2h", ev["home_team"], None))
        lines = [f"🏒 **{a} @ {h}** — {start}",
                 f"🥅 {ga['goalie']} vs {gh['goalie']} {gtag}" + ("" if kh and ka else " (⚠️ goalie w/ no data)"),
                 f"📊 Model: {h} {lh:.2f} – {a} {la:.2f} | {h} {p_home:.1%} ({p_to_am(p_home):+d})",
                 f"⚖️ Pinny fair: {h} {p_to_am(hf):+d} ({hf:.1%})" if hf else "⚖️ Pinny ML n/a"]
        notes = [n for n in (travel_note(a, ta), travel_note(h, th)) if n]
        if notes:
            lines.append("✈️ " + " | ".join(notes))
        if best:
            lines.append("```")
            for p in best[:8]:
                st = f"${p['stake']:.0f}" if p["stake"] else "—"
                lines.append(f"{p['tier']:<13}{p['lbl']:<12}{p['price']:+d} {p['book'][:10]:<10} "
                             f"fair {p['fair']:+d} | EV {p['pure']:+.1%}/{p['bev']:+.1%} | {st}")
            lines.append("```")
        else:
            lines.append("No book matches or beats Pinnacle fair.")
        post_discord("\n".join(lines))

        for p in best:
            log.append({"date": ds, "game_id": g["id"], "matchup": f"{a}@{h}", "tier": p["tier"],
                        "bet": p["lbl"], "market": p["mkey"], "side": p["side"],
                        "point": "" if p["point"] is None else p["point"], "book": p["book"], "price": p["price"], "fair_am": p["fair"],
                        "pinny_p": round(p["pf"], 4), "model_p": round(p["pm"], 4),
                        "ev_pure": round(p["pure"], 4), "ev_blend": round(p["bev"], 4),
                        "stake": round(p["stake"], 2), "home_goalie": gh["goalie"],
                        "away_goalie": ga["goalie"], "result": ""})
        state[str(g["id"])] = ds

    if log: log_rows(log)
    json.dump(state, open(STATE_FILE, "w"), indent=1)


if __name__ == "__main__":
    main()
