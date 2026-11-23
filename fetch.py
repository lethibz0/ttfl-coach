"""Collecte des données NBA depuis l'API publique ESPN.

- Calendrier + résultats (scoreboard par jour)
- Box scores détaillés + cotes (summary par match), mis en cache dans data/raw/
- Rapport de blessures courant

Usage : python fetch.py [--from 2025-10-21] [--to 2026-11-23]
"""
import argparse
import datetime as dt
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0 (ttfl-coach)"

# Périodes de saison régulière (playoffs / play-in exclus : TTFL = saison régulière)
SEASONS = {
    2025: ("2024-10-22", "2025-04-13"),
    2026: ("2025-10-21", "2026-04-12"),
    2027: ("2026-10-20", "2027-04-11"),
}


def get(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"échec {url} {params}")


def daterange(a, b):
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)


def scoreboard(day: dt.date):
    f = RAW / f"sb_{day:%Y%m%d}.json.gz"
    if f.exists():
        with gzip.open(f, "rt") as fh:
            return json.load(fh)
    d = get(f"{BASE}/scoreboard", {"dates": day.strftime("%Y%m%d"), "limit": 100}, tries=8)
    if day < dt.date.today() - dt.timedelta(days=1):  # jours passés : figés
        with gzip.open(f, "wt") as fh:
            json.dump(d, fh)
    return d


def parse_scoreboard(sb, day):
    rows = []
    for e in sb.get("events", []):
        c = e["competitions"][0]
        st = e["season"]["type"] if "season" in e else c.get("type", {}).get("id")
        teams = {t["homeAway"]: t for t in c["competitors"]}
        rows.append({
            "game_id": e["id"],
            "date": day.isoformat(),
            "start": e["date"],
            "season_type": e.get("season", {}).get("type", 2),
            "status": e["status"]["type"]["name"],
            "home_id": teams["home"]["team"]["id"],
            "home": teams["home"]["team"]["abbreviation"],
            "away_id": teams["away"]["team"]["id"],
            "away": teams["away"]["team"]["abbreviation"],
            "home_score": pd.to_numeric(teams["home"].get("score"), errors="coerce"),
            "away_score": pd.to_numeric(teams["away"].get("score"), errors="coerce"),
        })
    return rows


def summary(game_id, force=False):
    f = RAW / f"{game_id}.json.gz"
    if f.exists() and not force:
        with gzip.open(f, "rt") as fh:
            return json.load(fh)
    try:
        d = get(f"{BASE}/summary", {"event": game_id}, tries=6)
    except RuntimeError as e:
        print("  ignoré:", e)
        return {}
    keep = {k: d.get(k) for k in ("boxscore", "pickcenter", "header", "injuries")}
    if d.get("header", {}).get("competitions", [{}])[0].get("status", {}).get("type", {}).get("completed"):
        with gzip.open(f, "wt") as fh:
            json.dump(keep, fh)
    return keep


def _split(s):
    try:
        m, a = s.split("-")
        return int(m), int(a)
    except Exception:
        return 0, 0


def parse_box(game_id, d):
    rows = []
    for team in d.get("boxscore", {}).get("players", []) or []:
        tid = team["team"]["id"]
        for block in team["statistics"]:
            labels = block["labels"]
            for a in block["athletes"]:
                ath = a.get("athlete") or {}
                if "id" not in ath:
                    continue
                r = {
                    "game_id": game_id, "team_id": tid,
                    "player_id": ath["id"], "name": ath["displayName"],
                    "pos": (ath.get("position") or {}).get("abbreviation", ""),
                    "starter": bool(a.get("starter")),
                    "dnp": bool(a.get("didNotPlay")) or not a.get("stats"),
                    "dnp_reason": a.get("reason", ""),
                }
                st = dict(zip(labels, a.get("stats") or []))
                fgm, fga = _split(st.get("FG", "0-0"))
                tpm, tpa = _split(st.get("3PT", "0-0"))
                ftm, fta = _split(st.get("FT", "0-0"))
                num = lambda k: float(np.nan_to_num(pd.to_numeric(st.get(k, 0), errors="coerce")))
                r.update(min=num("MIN"), pts=num("PTS"), reb=num("REB"), ast=num("AST"),
                         stl=num("STL"), blk=num("BLK"), tov=num("TO"), oreb=num("OREB"),
                         pf=num("PF"), pm=num("+/-"),
                         fgm=fgm, fga=fga, tpm=tpm, tpa=tpa, ftm=ftm, fta=fta)
                rows.append(r)
    return rows


def parse_odds(game_id, d):
    pc = (d.get("pickcenter") or [])
    if not pc:
        return {"game_id": game_id}
    p = pc[0]
    home_fav = p.get("homeTeamOdds", {}).get("favorite")
    spread = p.get("spread")
    # spread_home : écart attendu du point de vue de l'équipe à domicile (négatif = favori)
    sh = None
    if spread is not None:
        sh = -abs(spread) if home_fav else abs(spread)
    return {"game_id": game_id, "total": p.get("overUnder"), "spread_home": sh}


def ttfl_score(df):
    """Barème TTFL : bonus (PTS+REB+AST+STL+BLK+FGM+3PM+FTM) - malus (TO + tirs ratés)."""
    bonus = df.pts + df.reb + df.ast + df.stl + df.blk + df.fgm + df.tpm + df.ftm
    malus = df.tov + (df.fga - df.fgm) + (df.tpa - df.tpm) + (df.fta - df.ftm)
    return bonus - malus


def game_row(game_id, d):
    h = d["header"]
    c = h["competitions"][0]
    t = {x["homeAway"]: x for x in c["competitors"]}
    start = pd.Timestamp(c["date"]).tz_convert("America/New_York")
    return {
        "game_id": game_id, "date": start.date().isoformat(), "start": c["date"],
        "season": h["season"]["year"], "season_type": h["season"]["type"],
        "home_id": t["home"]["team"]["id"], "home": t["home"]["team"]["abbreviation"],
        "away_id": t["away"]["team"]["id"], "away": t["away"]["team"]["abbreviation"],
        "home_score": pd.to_numeric(t["home"].get("score"), errors="coerce"),
        "away_score": pd.to_numeric(t["away"].get("score"), errors="coerce"),
    }


def update_cache(until: dt.date):
    """Télécharge (lentement, séquentiellement) les matchs terminés absents du cache."""
    cached = {f.name.split(".")[0] for f in RAW.glob("4*.json.gz")}
    last = SEASONS[2026][0]
    if cached:
        dates = []
        for gid in cached:
            with gzip.open(RAW / f"{gid}.json.gz", "rt") as fh:
                dates.append(game_row(gid, json.load(fh))["date"])
        last = max(dates)
    day = dt.date.fromisoformat(last) - dt.timedelta(days=1)  # marge de sécurité
    while day <= until:
        sb = scoreboard(day)
        for e in sb.get("events", []):
            if e["status"]["type"]["name"] == "STATUS_FINAL" and e["id"] not in cached:
                summary(e["id"])
                time.sleep(0.5)
        time.sleep(0.5)
        day += dt.timedelta(days=1)


def build_from_cache():
    rows, box, odds = [], [], []
    for f in sorted(RAW.glob("4*.json.gz")):
        gid = f.name.split(".")[0]
        with gzip.open(f, "rt") as fh:
            d = json.load(fh)
        g = game_row(gid, d)
        if g["season_type"] != 2 or int(g["home_id"]) > 100 or int(g["away_id"]) > 100:  # exclut All-Star
            continue
        rows.append(g)
        box += parse_box(gid, d)
        odds.append(parse_odds(gid, d))
    return pd.DataFrame(rows), pd.DataFrame(box), pd.DataFrame(odds)


def injuries():
    d = get(f"{BASE}/injuries")
    rows = []
    for t in d.get("injuries", []):
        for i in t.get("injuries", []):
            a = i.get("athlete", {})
            pid = None
            for l in a.get("links", []):
                if "/id/" in l.get("href", ""):
                    pid = l["href"].split("/id/")[1].split("/")[0]
                    break
            rows.append({"team": t.get("displayName"), "player_id": pid,
                         "name": a.get("displayName"), "status": i.get("status"),
                         "comment": i.get("shortComment"), "date": i.get("date")})
    return pd.DataFrame(rows)


def upcoming(days=30):
    """Calendrier (+ cotes si dispo) des prochains jours."""
    today = dt.date.today()
    rng = list(daterange(today, today + dt.timedelta(days=days)))
    with ThreadPoolExecutor(8) as ex:
        sbs = list(ex.map(scoreboard, rng))
    g = pd.DataFrame([r for sb, d in zip(sbs, rng) for r in parse_scoreboard(sb, d)])
    g = g[(g.season_type == 2) & (g.status != "STATUS_FINAL")]
    odds = []
    for sb in sbs:
        for e in sb.get("events", []):
            o = (e["competitions"][0].get("odds") or [{}])[0]
            if o:
                home_fav = o.get("homeTeamOdds", {}).get("favorite")
                sp = o.get("spread")
                odds.append({"game_id": e["id"], "total": o.get("overUnder"),
                             "spread_home": (-abs(sp) if home_fav else abs(sp)) if sp is not None else None})
    if odds:
        g = g.merge(pd.DataFrame(odds), on="game_id", how="left")
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update", action="store_true")
    a = ap.parse_args()
    if not a.no_update:
        update_cache(dt.date.today() - dt.timedelta(days=1))
    games, box, odds = build_from_cache()
    box["ttfl"] = ttfl_score(box)
    out = ROOT / "data"
    games.to_parquet(out / "games.parquet")
    box.to_parquet(out / "box.parquet")
    odds.to_parquet(out / "odds.parquet")
    print(games.groupby("season").size())
    print(box[~box.dnp].ttfl.describe())


def backfill(start: dt.date, end: dt.date, pause=0.8):
    """Téléchargement lent et poli d'une saison passée (séquentiel, pause entre requêtes)."""
    day = start
    while day <= end:
        try:
            sb = scoreboard(day)
        except RuntimeError:
            print("pause 60s (limitation ESPN)", day, flush=True)
            time.sleep(60)
            continue
        for e in sb.get("events", []):
            if e["status"]["type"]["name"] == "STATUS_FINAL" and not (RAW / f"{e['id']}.json.gz").exists():
                if not summary(e["id"]):
                    time.sleep(60)
                time.sleep(pause)
        time.sleep(pause)
        day += dt.timedelta(days=1)
    print("backfill terminé", flush=True)
