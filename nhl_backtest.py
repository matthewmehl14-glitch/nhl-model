#!/usr/bin/env python3
"""
NHL Backtest + Tuner
Point-in-time replay of nhl_model.py on past season(s). No look-ahead:
  - Team xGF/GF/xGA and goalie GSAx use ONLY games before each date, blended with the
    prior full season exactly like the live model.
  - Starter = goalie who faced the most shots for that team (stands in for "confirmed").
  - Data: MoneyPuck shot files (xG per shot) + NHL API finals (OT/SO settled like books).

Tunes:  HFA, XG_WEIGHT, PRIOR_GAMES, PRIOR_REGRESS, GOALIE_K, B2B_PENALTY,
        B2B_TRAVEL_K, TZ_K, DENSE_K (travel)  -> ML log loss
        ENG_PROB -> puck-line log loss;  OT_GOAL_PROB measured directly
Optional BT_ODDS=1: Pinnacle pre-game ML from Odds API historical (~10 credits/day, cached)
        -> is the model adding info beyond Pinnacle? best MODEL_WEIGHT.

Env: BT_SEASONS="2025" (season START years, e.g. "2024,2025"), BT_ODDS=0|1,
     ODDS_API_KEY, DISCORD_WEBHOOK_URL, SHOTS_URL (override if MoneyPuck moves the files)
"""
import os, io, csv, json, math, zipfile, datetime as dt
from collections import defaultdict
import numpy as np
import requests
import nhl_model as M

SEASONS = [int(s) for s in os.getenv("BT_SEASONS", "2025").split(",") if s.strip()]
USE_ODDS = os.getenv("BT_ODDS", "0") == "1"
ODDS_CACHE = "bt_odds_cache.json"
UA = {"User-Agent": "Mozilla/5.0 (nhl-backtest)"}
NHL_TEAMS = sorted(set(M.TEAMS.values()))
KMAX = 16

LIVE = dict(HFA=M.HFA, XG_WEIGHT=M.XG_WEIGHT, PRIOR_GAMES=M.PRIOR_GAMES, PRIOR_REGRESS=M.PRIOR_REGRESS,
            GOALIE_K=M.GOALIE_K, B2B_PENALTY=M.B2B_PENALTY, B2B_TRAVEL_K=M.B2B_TRAVEL_K, TZ_K=M.TZ_K,
            DENSE_K=M.DENSE_K, ENG_PROB=M.ENG_PROB, OT_GOAL_PROB=M.OT_GOAL_PROB)
GRID = dict(HFA=[1.0, 1.015, 1.025, 1.035, 1.05, 1.065],
            XG_WEIGHT=[0.4, 0.55, 0.7, 0.85, 1.0],
            PRIOR_GAMES=[3, 6, 10, 15, 25, 40],
            PRIOR_REGRESS=[0.15, 0.25, 0.33, 0.45, 0.6],
            GOALIE_K=[40, 80, 120, 200, 350, 600],
            B2B_PENALTY=[0.0, 0.03, 0.05, 0.07, 0.09, 0.11],
            B2B_TRAVEL_K=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05],
            TZ_K=[0.0, 0.005, 0.01, 0.015, 0.02, 0.03],
            DENSE_K=[0.0, 0.01, 0.02, 0.03, 0.05])
TRAVEL_KEYS = ("B2B_TRAVEL_K", "TZ_K", "DENSE_K")
ENG_GRID = [0.0, 0.05, 0.08, 0.11, 0.13, 0.16, 0.2, 0.25]


def one(v): return str(v).strip() in ("1", "1.0", "True", "true")
def fl(v):
    try: return float(v)
    except (TypeError, ValueError): return 0.0
def fix(code): return M.MP_FIX.get(code, code)


# ============================ DATA ============================
def load_shots(season):
    """Same loader the live model uses (nhl_model.load_shots)."""
    return M.load_shots(season)


def load_schedule(season):
    sid = f"{season}{season + 1}"
    out = {}
    for t in NHL_TEAMS:
        try:
            js = requests.get(f"https://api-web.nhle.com/v1/club-schedule-season/{t}/{sid}",
                              headers=UA, timeout=25).json()
        except Exception as e:
            print(f"[data] schedule {t} failed: {e}"); continue
        for g in js.get("games", []):
            if g.get("gameType") != 2 or g.get("gameState") not in ("OFF", "FINAL"): continue
            out[g["id"]] = dict(date=g["gameDate"], h=g["homeTeam"]["abbrev"], a=g["awayTeam"]["abbrev"],
                                hs=g["homeTeam"]["score"], as_=g["awayTeam"]["score"],
                                last=(g.get("gameOutcome") or {}).get("lastPeriodType", "REG"))
    print(f"[data] schedule {season}: {len(out)} final games")
    return out


def season_summary(shots, sched):
    """Full-season per-game team rates + goalie totals (used as the 'prior season')."""
    t = defaultdict(lambda: np.zeros(4))  # gp, xgf, gf, xga
    gl = defaultdict(lambda: np.zeros(2))  # xga, gsax
    for gid, g in shots.items():
        s = sched.get(gid)
        if not s: continue
        t[s["h"]] += [1, g["h_xg"], g["h_g"], g["a_xg"]]
        t[s["a"]] += [1, g["a_xg"], g["a_g"], g["h_xg"]]
        for (_, gk), (_, xga, ga) in g["goalies"].items():
            gl[gk] += [xga, xga - ga]
    tot = sum(t.values())
    lg = dict(xgf=tot[1] / tot[0], gf=tot[2] / tot[0], xga=tot[3] / tot[0])
    teams = {k: dict(xgf=v[1] / v[0], gf=v[2] / v[0], xga=v[3] / v[0]) for k, v in t.items() if v[0]}
    return teams, dict(gl), lg


def build_features(season):
    prev_shots, prev_sched = load_shots(season - 1), load_schedule(season - 1)
    p_team, p_gl, p_lg = season_summary(prev_shots, prev_sched)
    shots, sched = load_shots(season), load_schedule(season)

    cur_t = defaultdict(lambda: np.zeros(4))
    cur_g = defaultdict(lambda: np.zeros(2))
    hist = defaultdict(list)  # team -> [(date, location)] from the full schedule
    for s_ in sched.values():
        hist[s_["h"]].append((s_["date"], s_["h"])); hist[s_["a"]].append((s_["date"], s_["h"]))
    for v in hist.values(): v.sort()
    rows = []
    by_date = defaultdict(list)
    for gid, s in sched.items():
        if gid in shots: by_date[s["date"]].append(gid)

    for d in sorted(by_date):
        tot = sum(cur_t.values()) if cur_t else np.zeros(4)
        n_teams = sum(1 for v in cur_t.values() if v[0] > 0)
        for gid in by_date[d]:
            s, g = sched[gid], shots[gid]
            row = dict(gid=gid, date=d, h=s["h"], a=s["a"], hs=s["hs"], as_=s["as_"], last=s["last"],
                       lg_c_teams=n_teams,
                       lg_c_xgf=tot[1] / tot[0] if tot[0] else 0, lg_c_gf=tot[2] / tot[0] if tot[0] else 0,
                       lg_c_xga=tot[3] / tot[0] if tot[0] else 0,
                       lg_p_xgf=p_lg["xgf"], lg_p_gf=p_lg["gf"], lg_p_xga=p_lg["xga"])
            for side, team in (("h", s["h"]), ("a", s["a"])):
                c = cur_t[team]; p = p_team.get(team, p_lg)
                starters = [(v[0], k[1]) for k, v in g["goalies"].items() if k[0] == side]
                gk = max(starters)[1] if starters else 0
                cg, pg = cur_g.get(gk, np.zeros(2)), p_gl.get(gk, np.zeros(2))
                row.update({f"{side}_cgp": c[0], f"{side}_cxgf": c[1], f"{side}_cgf": c[2], f"{side}_cxga": c[3],
                            f"{side}_pxgf": p["xgf"], f"{side}_pgf": p["gf"], f"{side}_pxga": p["xga"],
                            f"{side}_gid": gk, f"{side}_g_cxga": cg[0], f"{side}_g_cgsax": cg[1],
                            f"{side}_g_pxga": pg[0], f"{side}_g_pgsax": pg[1],
                            })
                tv = M.travel_features([x for x in hist[team] if x[0] < d], s["h"], d)
                row.update({f"{side}_{k}": v for k, v in tv.items()})
            rows.append(row)
        for gid in by_date[d]:  # update AFTER the whole date -> no same-day leakage
            s, g = sched[gid], shots[gid]
            cur_t[s["h"]] += [1, g["h_xg"], g["h_g"], g["a_xg"]]
            cur_t[s["a"]] += [1, g["a_xg"], g["a_g"], g["h_xg"]]
            for (_, gk), (_, xga, ga) in g["goalies"].items():
                cur_g[gk] = cur_g.get(gk, np.zeros(2)) + [xga, xga - ga]
    print(f"[features] {season}: {len(rows)} games")
    return rows


# ============================ MODEL (vectorized mirror of live) ============================
def lambdas(F, P):
    xw, R, PG = P["XG_WEIGHT"], P["PRIOR_REGRESS"], P["PRIOR_GAMES"]
    ok = F["lg_c_teams"] >= 30
    lg_off = np.where(ok, xw * F["lg_c_xgf"] + (1 - xw) * F["lg_c_gf"], xw * F["lg_p_xgf"] + (1 - xw) * F["lg_p_gf"])
    lg_def = np.where(ok, F["lg_c_xga"], F["lg_p_xga"])
    lg_goals = np.where(ok, F["lg_c_gf"], F["lg_p_gf"])
    r = {}
    for s in ("h", "a"):
        gp = F[f"{s}_cgp"]; w = gp / (gp + PG); safe = np.maximum(gp, 1)
        po = (xw * F[f"{s}_pxgf"] + (1 - xw) * F[f"{s}_pgf"]) * (1 - R) + lg_off * R
        pd = F[f"{s}_pxga"] * (1 - R) + lg_def * R
        co = (xw * F[f"{s}_cxgf"] + (1 - xw) * F[f"{s}_cgf"]) / safe
        cd = F[f"{s}_cxga"] / safe
        r[s + "_off"] = np.where(gp > 0, w * co + (1 - w) * po, po) / lg_off
        r[s + "_def"] = np.where(gp > 0, w * cd + (1 - w) * pd, pd) / lg_def
        xga = F[f"{s}_g_cxga"] + 0.6 * F[f"{s}_g_pxga"]
        gsax = F[f"{s}_g_cgsax"] + 0.6 * F[f"{s}_g_pgsax"]
        r[s + "_gk"] = np.where(xga > 0, np.clip(1 - gsax / (xga + P["GOALIE_K"]), 0.85, 1.12), M.UNKNOWN_GOALIE)
    lh = lg_goals * r["h_off"] * r["a_def"] * P["HFA"] * r["a_gk"]
    la = lg_goals * r["a_off"] * r["h_def"] / P["HFA"] * r["h_gk"]
    fat = {}
    for s in ("h", "a"):  # vectorized mirror of nhl_model.fatigue()
        f = (P["B2B_PENALTY"] + P["B2B_TRAVEL_K"] * F[f"{s}_miles"] / 1000) * F[f"{s}_b2b"] \
            + P["TZ_K"] * F[f"{s}_tz"] * (F[f"{s}_rest"] <= 2) + P["DENSE_K"] * F[f"{s}_dense"]
        fat[s] = np.clip(f, 0, M.FATIGUE_CAP)
    lh = lh * (1 - fat["h"]) * (1 + fat["a"])
    la = la * (1 - fat["a"]) * (1 + fat["h"])
    return lh, la


FACT = np.array([math.factorial(k) for k in range(KMAX)], dtype=float)
def pmf(l): return np.exp(-l[:, None]) * l[:, None] ** np.arange(KMAX) / FACT
HOME_MASK = np.tril(np.ones((KMAX, KMAX)), -1).astype(bool)


def p_home_ml(lh, la, otg):
    J = pmf(lh)[:, :, None] * pmf(la)[:, None, :]
    reg = J[:, HOME_MASK].sum(1)
    tie = np.einsum("gii->g", J)
    return reg + tie * (otg * lh / (lh + la) + (1 - otg) * 0.5)


def final_dist(lh, la, eng, otg):
    """Same settlement rules as the live sim: ENG on 1-goal leads, OT/SO winner +1."""
    J = pmf(lh)[:, :, None] * pmf(la)[:, None, :]
    F = np.zeros((len(lh), KMAX + 1, KMAX + 1))
    pt = otg * lh / (lh + la) + (1 - otg) * 0.5
    for i in range(KMAX):
        for j in range(KMAX):
            m = J[:, i, j]
            if i - j == 1:   F[:, i, j] += m * (1 - eng); F[:, i + 1, j] += m * eng
            elif j - i == 1: F[:, i, j] += m * (1 - eng); F[:, i, j + 1] += m * eng
            elif i == j:     F[:, i + 1, j] += m * pt;    F[:, i, j + 1] += m * (1 - pt)
            else:            F[:, i, j] += m
    return F


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ============================ ODDS (optional) ============================
def pinnacle_history(dates):
    cache = json.load(open(ODDS_CACHE)) if os.path.exists(ODDS_CACHE) else {}
    need = [d for d in dates if d not in cache]
    print(f"[odds] {len(need)} dates to fetch (~{len(need) * 10} credits), {len(dates) - len(need)} cached")
    for d in need:
        try:
            r = requests.get("https://api.the-odds-api.com/v4/historical/sports/icehockey_nhl/odds",
                             params={"apiKey": M.ODDS_API_KEY, "bookmakers": "pinnacle", "markets": "h2h",
                                     "oddsFormat": "american", "date": f"{d}T22:00:00Z"}, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"[odds] {d} failed: {e}"); continue
        day = {}
        for ev in r.json().get("data", []):
            h, a = M.team_abbr(ev["home_team"]), M.team_abbr(ev["away_team"])
            pin = next((b for b in ev.get("bookmakers", []) if b["key"] == "pinnacle"), None)
            if not (h and a and pin): continue
            if not pin.get("markets"): continue
            outs = pin["markets"][0]["outcomes"]
            if len(outs) != 2: continue
            probs = M.power_devig([1 / M.am_to_dec(o["price"]) for o in outs])
            ph = next(p for o, p in zip(outs, probs) if o["name"] == ev["home_team"])
            day[f"{a}@{h}"] = round(ph, 5)
        cache[d] = day
        json.dump(cache, open(ODDS_CACHE, "w"))
    return cache


# ============================ MAIN ============================
def main():
    rows = [r for s in SEASONS for r in build_features(s)]
    keys = [k for k in rows[0] if k not in ("date", "h", "a", "last")]
    F = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    y = (F["hs"] > F["as_"]).astype(float)
    n = len(y)

    # ---- OT/SO rate measured directly ----
    ot = sum(r["last"] == "OT" for r in rows); so = sum(r["last"] == "SO" for r in rows)
    otg_emp = ot / (ot + so) if ot + so else M.OT_GOAL_PROB

    def score(P): lh, la = lambdas(F, P); return ll(p_home_ml(lh, la, P["OT_GOAL_PROB"]), y)

    base_ll = ll(np.full(n, y.mean()), y)
    live_ll = score(LIVE)

    # ---- coordinate descent on ML log loss ----
    best = dict(LIVE, OT_GOAL_PROB=round(otg_emp, 3))
    best_ll = score(best)
    for _ in range(3):
        moved = False
        for k, vals in GRID.items():
            for v in vals:
                trial = dict(best, **{k: v})
                s = score(trial)
                if s < best_ll - 1e-5:
                    best, best_ll, moved = trial, s, True
        if not moved: break

    no_travel_ll = score(dict(best, **{k: 0.0 for k in TRAVEL_KEYS}))
    travel_gain = no_travel_ll - best_ll
    lh, la = lambdas(F, best)
    p_ml = p_home_ml(lh, la, best["OT_GOAL_PROB"])
    margin = F["hs"] - F["as_"]; total = F["hs"] + F["as_"]

    # ---- ENG on puck line (favorite -1.5) ----
    fav_home = p_ml >= 0.5
    y_pl = np.where(fav_home, margin >= 2, margin <= -2).astype(float)
    i_idx, j_idx = np.meshgrid(np.arange(KMAX + 1), np.arange(KMAX + 1), indexing="ij")
    def pl_probs(eng):
        D = final_dist(lh, la, eng, best["OT_GOAL_PROB"])
        ph2 = (D * (i_idx - j_idx >= 2)).sum((1, 2)); pa2 = (D * (j_idx - i_idx >= 2)).sum((1, 2))
        return np.where(fav_home, ph2, pa2), D
    eng_scores = {e: ll(pl_probs(e)[0], y_pl) for e in ENG_GRID}
    best["ENG_PROB"] = min(eng_scores, key=eng_scores.get)
    p_pl, D = pl_probs(best["ENG_PROB"])

    tot_grid = i_idx + j_idx
    p_o55 = (D * (tot_grid > 5.5)).sum((1, 2)); p_o65 = (D * (tot_grid > 6.5)).sum((1, 2))
    exp_total = (D * tot_grid).sum((1, 2))

    def calib(p, yy, edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() >= 15: out.append((f"{lo:.2f}-{hi:.2f}", int(m.sum()), float(p[m].mean()), float(yy[m].mean())))
        return out
    cal_ml = calib(p_ml, y, [0, .35, .42, .48, .52, .58, .65, 1.01])

    res = dict(seasons=SEASONS, games=n, home_win_rate=float(y.mean()),
               ml=dict(baseline_ll=base_ll, live_params_ll=live_ll, tuned_ll=best_ll,
                       brier=float(np.mean((p_ml - y) ** 2)), accuracy=float(np.mean((p_ml >= .5) == y)),
                       no_travel_ll=no_travel_ll, travel_gain=travel_gain),
               puckline=dict(fav_cover_rate=float(y_pl.mean()), model_avg=float(p_pl.mean()),
                             ll=ll(p_pl, y_pl), eng_scores=eng_scores),
               totals=dict(actual_avg=float(total.mean()), model_avg=float(exp_total.mean()),
                           over55_actual=float((total > 5.5).mean()), over55_model=float(p_o55.mean()),
                           over65_actual=float((total > 6.5).mean()), over65_model=float(p_o65.mean()),
                           over55_ll=ll(p_o55, (total > 5.5).astype(float)),
                           over65_ll=ll(p_o65, (total > 6.5).astype(float))),
               ot_goal_prob_measured=otg_emp, tuned_params=best, calibration_ml=cal_ml)

    # ---- optional: Pinnacle comparison ----
    pin = np.full(n, np.nan)
    if USE_ODDS and M.ODDS_API_KEY:
        cache = pinnacle_history(sorted({r["date"] for r in rows}))
        for i, r in enumerate(rows):
            v = cache.get(r["date"], {}).get(f"{r['a']}@{r['h']}")
            if v is not None: pin[i] = v
        m = ~np.isnan(pin)
        if m.sum() > 50:
            blends = {round(w, 2): ll((1 - w) * pin[m] + w * p_ml[m], y[m]) for w in np.arange(0, 0.65, 0.05)}
            bw = min(blends, key=blends.get)
            res["pinnacle"] = dict(matched=int(m.sum()), pinnacle_ll=ll(pin[m], y[m]), model_ll=ll(p_ml[m], y[m]),
                                   best_model_weight=bw, best_blend_ll=blends[bw], blends=blends,
                                   corr=float(np.corrcoef(pin[m], p_ml[m])[0, 1]))
            best["MODEL_WEIGHT"] = bw

    # ---- write outputs ----
    json.dump(res, open("backtest_results.json", "w"), indent=1, default=float)
    with open("backtest_games.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "matchup", "home_goalie_id", "away_goalie_id", "lam_home", "lam_away",
                    "p_home_ml", "pinny_home", "p_fav_m1.5", "exp_total", "home_score", "away_score", "last_period"])
        for i, r in enumerate(rows):
            w.writerow([r["date"], f"{r['a']}@{r['h']}", r["h_gid"], r["a_gid"], f"{lh[i]:.3f}", f"{la[i]:.3f}",
                        f"{p_ml[i]:.4f}", "" if np.isnan(pin[i]) else f"{pin[i]:.4f}", f"{p_pl[i]:.4f}",
                        f"{exp_total[i]:.2f}", r["hs"], r["as_"], r["last"]])

    # ---- Discord summary ----
    t = res["totals"]; pl = res["puckline"]; ml = res["ml"]
    L = [f"🧪 **NHL Backtest — seasons {', '.join(f'{s}-{str(s + 1)[2:]}' for s in SEASONS)}** ({n} games)", "```",
         f"ML log loss   baseline {ml['baseline_ll']:.4f} | live {ml['live_params_ll']:.4f} | tuned {ml['tuned_ll']:.4f}",
         f"ML Brier {ml['brier']:.4f}  acc {ml['accuracy']:.1%}  home win {res['home_win_rate']:.1%}",
         f"Travel       without {no_travel_ll:.4f} | with {best_ll:.4f} | gain {travel_gain:.4f}",
         f"PL fav -1.5  actual {pl['fav_cover_rate']:.1%} | model {pl['model_avg']:.1%}",
         f"Totals avg   actual {t['actual_avg']:.2f} | model {t['model_avg']:.2f}",
         f"Over 5.5     actual {t['over55_actual']:.1%} | model {t['over55_model']:.1%}",
         f"Over 6.5     actual {t['over65_actual']:.1%} | model {t['over65_model']:.1%}",
         f"OT decided before SO: {otg_emp:.1%}", "", "ML CALIBRATION  bucket   n   pred  actual"]
    for b, cnt, pp, aa in cal_ml: L.append(f"                {b} {cnt:>4} {pp:.1%} {aa:.1%}")
    if "pinnacle" in res:
        pz = res["pinnacle"]
        L += ["", f"VS PINNACLE (n={pz['matched']})  pinny {pz['pinnacle_ll']:.4f} | model {pz['model_ll']:.4f} | "
                  f"best blend {pz['best_blend_ll']:.4f} @ w={pz['best_model_weight']}  corr {pz['corr']:.2f}"]
    L += ["", "PASTE INTO nhl_model.py CONFIG:"]
    for k, v in best.items(): L.append(f"{k:<14}= {v}")
    L.append("```")
    if ml["tuned_ll"] >= ml["baseline_ll"]:
        L.append("⚠️ Model is not beating a home-rate baseline — do not bet model-tiered plays yet.")
    if travel_gain < 0.0005:
        L.append("ℹ️ Travel adds little/no predictive value here — the tuned travel values may be 0, which is fine.")
    if "pinnacle" in res and res["pinnacle"]["best_model_weight"] == 0:
        L.append("⚠️ Model adds no info beyond Pinnacle — set MODEL_WEIGHT=0, treat output as pure line shopping.")
    M.post_discord("\n".join(L))


if __name__ == "__main__":
    main()
