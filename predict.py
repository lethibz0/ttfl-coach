"""Pipeline quotidien : met à jour les données, projette les matchs du soir et
des 14 prochains jours, écrit site/data.json pour l'interface.

Usage : python predict.py [--no-update] [--date 2026-11-23]
"""
import argparse
import datetime as dt
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import fetch
import model as M
import availability as A

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SITE = ROOT / "site"
HORIZON = 14
TEAM_INFO = {}


def team_names(games):
    m = {}
    for _, g in games.iterrows():
        m[str(g.home_id)] = g.home
        m[str(g.away_id)] = g.away
    return m


def rosters(box, games, today):
    """Rotation actuelle : joueurs ayant joué dans les 15 derniers jours, équipe la plus récente."""
    b = box.merge(games[["game_id", "date"]], on="game_id")
    b["date"] = pd.to_datetime(b.date)
    b = b[(~b.dnp) & (b["min"] > 0)]
    last = b.sort_values("date").groupby("player_id").tail(1)
    last = last[last.date >= pd.Timestamp(today) - pd.Timedelta(days=15)]
    return last[["player_id", "name", "pos", "team_id"]]


def main(today: dt.date, update=True):
    if update:
        fetch.update_cache(today - dt.timedelta(days=1))
    games, box, odds = fetch.build_from_cache()
    box["ttfl"] = fetch.ttfl_score(box)
    games = games.merge(odds, on="game_id", how="left")

    # --- matchs à venir (ce soir + horizon) ---
    up = []
    for k in range(HORIZON + 1):
        day = today + dt.timedelta(days=k)
        sb = fetch.scoreboard(day)
        for r in fetch.parse_scoreboard(sb, day):
            if r["season_type"] != 2 or int(r["home_id"]) > 100:
                continue
            e = next(e for e in sb["events"] if e["id"] == r["game_id"])
            o = (e["competitions"][0].get("odds") or [{}])[0]
            sp = o.get("spread")
            r["total"] = o.get("overUnder")
            r["spread_home"] = (-abs(sp) if o.get("homeTeamOdds", {}).get("favorite") else abs(sp)) if sp is not None else None
            r["season"] = 2027
            up.append(r)
    up = pd.DataFrame(up)
    up = up[~up.game_id.isin(games.game_id)]
    tonight_ids = set(up[up.date == today.isoformat()].game_id)
    all_games = pd.concat([games, up[games.columns.intersection(up.columns)]], ignore_index=True)

    # --- blessures ---
    inj = fetch.injuries()
    inj_status = dict(zip(inj.player_id.astype(str), inj.status))
    inj_comment = dict(zip(inj.player_id.astype(str), inj.comment))
    inj_date = dict(zip(inj.player_id.astype(str), inj.date))

    # --- lignes futures ---
    ros = rosters(box, games, today)
    fut = []
    for _, g in up.iterrows():
        gdate = dt.date.fromisoformat(g.date)
        for tid in (g.home_id, g.away_id):
            for _, p in ros[ros.team_id == tid].iterrows():
                st = inj_status.get(str(p.player_id))
                if st == "Out" and (gdate - today).days <= 3:
                    continue  # absent : ne joue pas (compte comme coéquipier absent)
                fut.append({"game_id": g.game_id, "team_id": tid, "player_id": p.player_id,
                            "name": p["name"], "pos": p.pos})
    fut = pd.DataFrame(fut)
    df = F.build(box, all_games, future=fut)
    ms = pickle.load(open(M.MODELS, "rb"))

    # --- probabilité de jouer ce soir ---
    tonight_games = up[up.game_id.isin(tonight_ids)]
    pan = A.panel(box, pd.concat([games, tonight_games[games.columns.intersection(tonight_games.columns)]]))
    pan = pan[pan.game_id.isin(tonight_ids)]
    pan["p_model"] = ms["avail"].predict_proba(pan[A.FEATS])[:, 1]
    p_model = dict(zip(pan.player_id, pan.p_model))
    last_start = games.groupby("home_id").start.max().combine(games.groupby("away_id").start.max(), max)
    f = M.predict(ms, df[df._future].copy())
    tn = team_names(all_games)

    # rang défensif adverse par poste (1 = défense la plus permissive)
    f["opp_rank"] = f.groupby(["date", "pg"]).opp_def.rank(ascending=False, method="min")

    # historique récent par joueur pour la fiche
    hist = box.merge(games[["game_id", "date", "home_id", "away_id"]], on="game_id")
    hist = hist[~hist.dnp].sort_values("date")
    hist["opp"] = np.where(hist.team_id == hist.home_id, hist.away_id, hist.home_id)
    last5 = {pid: [{"d": r.date[5:], "o": tn.get(str(r.opp), "?"), "s": int(r.ttfl), "m": int(np.nan_to_num(getattr(r, "min")))}
                   for r in h.tail(5).itertuples()] for pid, h in hist.groupby("player_id")}
    season_avg = hist[hist.date >= fetch.SEASONS[2027][0]].groupby("player_id").ttfl.agg(["mean", "size"])

    # absents de la rotation par équipe ce soir (noms)
    out_by_team = {}
    for pid, st in inj_status.items():
        if st == "Out":
            r = ros[ros.player_id == pid]
            if len(r):
                out_by_team.setdefault(r.team_id.iloc[0], []).append(r.name.iloc[0])

    players = []
    for pid, grp in f.sort_values("date").groupby("player_id"):
        tonight_row = grp[grp.game_id.isin(tonight_ids)]
        future = grp[~grp.game_id.isin(tonight_ids)]
        sched = [{"d": r.date.strftime("%Y-%m-%d"), "o": tn.get(str(r.opp_id), "?"), "h": int(r.home),
                  "p": round(float(r.proj), 1)} for r in future.itertuples()]
        entry = {
            "id": pid, "name": grp.name.iloc[0], "team": tn.get(str(grp.team_id.iloc[0]), "?"),
            "pos": grp.pos.iloc[0], "base": round(float(grp.base.iloc[0]), 1),
            "avg": round(float(season_avg["mean"].get(pid, np.nan)), 1) if pid in season_avg.index else None,
            "gp": int(season_avg["size"].get(pid, 0)) if pid in season_avg.index else 0,
            "last5": last5.get(pid, []), "future": sched,
            "status": inj_status.get(str(pid)), "comment": inj_comment.get(str(pid)),
        }
        if len(tonight_row):
            r = tonight_row.iloc[0]
            pm = float(p_model.get(pid, 0.85))
            ps, stale = A.status_prob(inj_status.get(str(pid)), inj_comment.get(str(pid)), today)
            pp = A.combine(pm, ps)
            entry["tonight"] = {
                "p_play": round(pp, 2), "exp": round(float(r.proj) * pp, 1), "stale": bool(stale),
                "opp": tn.get(str(r.opp_id), "?"), "home": int(r.home),
                "proj": round(float(r.proj), 1), "p10": round(float(r.p10), 1), "p90": round(float(r.p90), 1),
                "adj": round(float(r.adj), 1), "b2b": int(r.b2b), "rest": int(r.rest),
                "opp_rank": None if pd.isna(r.opp_rank) else int(r.opp_rank),
                "spread": None if pd.isna(r.spread) else float(r.spread),
                "total": None if pd.isna(r.total) else float(r.total),
                "miss": out_by_team.get(r.team_id, []),
                "min_l3": None if pd.isna(r.min_l3) else round(float(r.min_l3), 1),
            }
        players.append(entry)

    games_tonight = [{"id": g.game_id, "home": g.home, "away": g.away, "start": g.start,
                      "total": None if pd.isna(g.total) else g.total,
                      "spread_home": None if pd.isna(g.spread_home) else g.spread_home}
                     for g in up[up.game_id.isin(tonight_ids)].itertuples()]
    out_players = [{"name": n, "team": tn.get(str(ros[ros.name == n].team_id.iloc[0]), "?") if (ros.name == n).any() else "?",
                    "status": s, "comment": inj_comment.get(pid)}
                   for pid, s in inj_status.items() for n in [inj[inj.player_id.astype(str) == pid].name.iloc[0]]]
    # joueurs de la rotation déclarés Out alors que leur équipe joue ce soir
    tg_tonight = {}
    for g in up[up.game_id.isin(tonight_ids)].itertuples():
        tg_tonight[g.home_id] = (tn.get(str(g.away_id)), 1)
        tg_tonight[g.away_id] = (tn.get(str(g.home_id)), 0)
    base_by_pid = df.drop_duplicates("player_id", keep="last").set_index("player_id").base
    out_tonight = []
    for _, r in ros.iterrows():
        st = inj_status.get(str(r.player_id))
        if st == "Out" and r.team_id in tg_tonight:
            out_tonight.append({"id": r.player_id, "name": r["name"], "team": tn.get(str(r.team_id)),
                                "opp": tg_tonight[r.team_id][0], "home": tg_tonight[r.team_id][1],
                                "base": round(float(base_by_pid.get(r.player_id, np.nan)), 1),
                                "comment": inj_comment.get(str(r.player_id))})
    out_tonight.sort(key=lambda x: -(x["base"] if x["base"] == x["base"] else 0))
    data = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
            "date": today.isoformat(), "games": games_tonight, "players": players,
            "out_tonight": out_tonight, "injuries": out_players}
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str))
    top = sorted([p for p in players if "tonight" in p], key=lambda p: -p["tonight"]["exp"])[:12]
    print(f"{len(games_tonight)} matchs ce soir, {sum('tonight' in p for p in players)} joueurs projetés")
    for p in top:
        t = p["tonight"]
        print(f"  {p['name']:<26} {p['team']:>4} vs {t['opp']:<4} espéré {t['exp']:5.1f} (proj {t['proj']:5.1f} x {t['p_play']:.0%})  [{t['p10']:.0f}-{t['p90']:.0f}]  adj {t['adj']:+.1f}  {p['status'] or ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--date", default=None)
    a = ap.parse_args()
    today = dt.date.fromisoformat(a.date) if a.date else pd.Timestamp.now(tz="America/New_York").date()
    main(today, update=not a.no_update)
