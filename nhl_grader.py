#!/usr/bin/env python3
"""
NHL Grader
- Grades ungraded rows in nhl_plays_log.csv from NHL API final scores
  (final score includes the +1 OT/SO winner goal = sportsbook settlement)
- Writes result / units (flat 1u) / stake_profit back to the CSV
- Posts report to Discord: overall, by tier, model-filter test, market, EV bucket, book, calibration
Env: DISCORD_WEBHOOK_URL
"""
import os, csv, datetime as dt
from collections import defaultdict
from zoneinfo import ZoneInfo
import requests

LOG_FILE = "nhl_plays_log.csv"
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "")
ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (nhl-grader)"}


def am_to_dec(a): a = float(a); return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def finals(date):
    r = requests.get(f"https://api-web.nhle.com/v1/score/{date}", headers=UA, timeout=25)
    r.raise_for_status()
    return {str(g["id"]): g for g in r.json().get("games", []) if g.get("gameState") in ("OFF", "FINAL")}


def parse_bet(row):
    """Prefer structured columns; fall back to parsing the label ('TOR ML', 'TOR -1.5', 'Over 6.5')."""
    if row.get("market"):
        pt = float(row["point"]) if row.get("point") not in (None, "") else None
        return row["market"], row["side"], pt
    parts = row["bet"].split()
    if parts[0] in ("Over", "Under"): return "totals", parts[0], float(parts[1])
    if parts[1] == "ML": return "h2h", parts[0], None
    return "spreads", parts[0], float(parts[1])


def grade(row, g):
    h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
    hs, as_ = g["homeTeam"]["score"], g["awayTeam"]["score"]
    mkt, side, pt = parse_bet(row)
    if mkt == "totals":
        diff = (hs + as_ - pt) if side == "Over" else (pt - hs - as_)
    else:
        if side not in (h, a): return None
        margin = hs - as_ if side == h else as_ - hs
        diff = margin if mkt == "h2h" else margin + pt
    return "W" if diff > 0 else "L" if diff < 0 else "P"


def rec(rows):
    w = sum(r["result"] == "W" for r in rows); l = sum(r["result"] == "L" for r in rows)
    p = sum(r["result"] == "P" for r in rows)
    u = sum(float(r["units"]) for r in rows)
    risk = sum(1 for r in rows if r["result"] != "P")
    roi = u / risk if risk else 0
    return f"{w}-{l}" + (f"-{p}" if p else "") + f"  {u:+.2f}u  ROI {roi:+.1%}"


def ev_bucket(ev):
    ev = float(ev)
    return "0-1%" if ev < .01 else "1-2%" if ev < .02 else "2-4%" if ev < .04 else "4%+"


def main():
    if not os.path.exists(LOG_FILE):
        print("No log yet."); return
    with open(LOG_FILE, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys()) if rows else []
    for c in ("result", "units", "stake_profit"):
        if c not in fields: fields.append(c)

    today = dt.datetime.now(ET).date().isoformat()
    todo = sorted({r["date"] for r in rows if not r.get("result") and r["date"] <= today})
    newly = []
    for d in todo:
        try: fin = finals(d)
        except Exception as e:
            print(f"[grader] {d} fetch failed: {e}"); continue
        for r in rows:
            if r["date"] != d or r.get("result") or r["game_id"] not in fin: continue
            res = grade(r, fin[r["game_id"]])
            if not res: continue
            dec = am_to_dec(r["price"]); stake = float(r.get("stake") or 0)
            r["result"] = res
            r["units"] = f"{(dec - 1) if res == 'W' else -1 if res == 'L' else 0:.3f}"
            r["stake_profit"] = f"{stake * (dec - 1) if res == 'W' else -stake if res == 'L' else 0:.2f}"
            newly.append(r)

    with open(LOG_FILE, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in fields})

    graded = [r for r in rows if r.get("result") in ("W", "L", "P")]
    if not graded:
        print("Nothing graded yet."); return

    L = ["📋 **NHL Model — Grader Report**", "```"]
    if newly:
        L.append(f"NEW ({len(newly)})")
        for r in newly:
            L.append(f" {r['result']} {r['matchup']:<9}{r['bet']:<11}{int(float(r['price'])):+d} {r['book'][:9]}  {float(r['units']):+.2f}u")
        L.append("")
    L.append(f"ALL-TIME (flat 1u)  {rec(graded)}")
    staked = [r for r in graded if float(r.get("stake") or 0) > 0]
    if staked:
        risk = sum(float(r["stake"]) for r in staked if r["result"] != "P")
        prof = sum(float(r["stake_profit"]) for r in staked)
        L.append(f"KELLY STAKES        ${prof:+.2f} on ${risk:.0f}  ROI {prof / risk if risk else 0:+.1%}")

    def section(title, keyfn):
        groups = defaultdict(list)
        for r in graded: groups[keyfn(r)].append(r)
        L.append(f"\n{title}")
        for k in sorted(groups): L.append(f" {k:<14}{rec(groups[k])}")

    section("BY TIER", lambda r: r["tier"].split(" ", 1)[-1])
    section("MODEL FILTER", lambda r: "price-only" if "price-only" in r["tier"] else "model agrees")
    section("BY MARKET", lambda r: parse_bet(r)[0])
    section("BY PURE EV", lambda r: ev_bucket(r["ev_pure"]))
    section("BY BOOK", lambda r: r["book"][:13])

    dec_rows = [r for r in graded if r["result"] != "P"]
    if dec_rows:
        n = len(dec_rows)
        act = sum(r["result"] == "W" for r in dec_rows) / n
        pin = sum(float(r["pinny_p"]) for r in dec_rows) / n
        mod = sum(float(r["model_p"]) for r in dec_rows) / n
        L.append(f"\nCALIBRATION (n={n})  actual {act:.1%} | pinny {pin:.1%} | model {mod:.1%}")
    L.append("```")
    L.append("_Small samples lie. Judge the model filter after 300+ graded plays; CLV in Pikkit is the faster signal._")

    msg = "\n".join(L)
    if DISCORD_WEBHOOK:
        chunks, cur = [], ""
        for line in msg.split("\n"):
            if len(cur) + len(line) > 1850:
                chunks.append(cur + ("\n```" if cur.count("```") % 2 else "")); cur = "```\n" if cur.count("```") % 2 else ""
            cur += line + "\n"
        chunks.append(cur)
        for c in chunks: requests.post(DISCORD_WEBHOOK, json={"content": c}, timeout=20)
    else:
        print(msg)


if __name__ == "__main__":
    main()
