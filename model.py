"""Entraînement + évaluation (backtest) du modèle de projection TTFL."""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import features as F

ROOT = Path(__file__).parent
MODELS = ROOT / "data" / "models.pkl"


def make_models():
    common = dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=150,
                  l2_regularization=5.0, random_state=0)
    return {
        "mean": HistGradientBoostingRegressor(loss="squared_error", **common),
        "q10": HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **common),
        "q90": HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **common),
    }


def trainable(df):
    return df[(~df.dnp) & (~df._future) & (df["min"] > 0) & (df.gp >= 1)]


def fit(df):
    d = trainable(df)
    cols = [c for c in F.CONTEXT if d[c].notna().any()]
    ms = make_models()
    for m in ms.values():
        m.fit(d[cols], d.ttfl - d.base)
    ms["cols"] = cols
    return ms


def predict(ms, df):
    out = df.copy()
    X = df[ms["cols"]]
    out["adj"] = ms["mean"].predict(X)
    out["proj"] = out.base + out.adj
    out["p10"] = out.base + ms["q10"].predict(X)
    out["p90"] = out.base + ms["q90"].predict(X)
    return out


def simulate_season(test, score_col, cooldown=30):
    """Pick glouton chaque soir (meilleur score prévu non utilisé depuis 30 j).
    Disponibilité supposée connue (joueurs ayant effectivement joué)."""
    last_used = {}
    total, picks = 0.0, []
    for d, night in test.sort_values("date").groupby("date"):
        night = night.sort_values(score_col, ascending=False)
        for _, r in night.iterrows():
            lu = last_used.get(r.player_id)
            if lu is None or (d - lu).days > cooldown:
                last_used[r.player_id] = d
                total += r.ttfl
                picks.append((d.date().isoformat(), r["name"], round(r[score_col], 1), r.ttfl))
                break
    return total, picks


def walk_forward(df, periods, step_days=14):
    """Réentraîne le modèle toutes les 2 semaines avec les seules données passées."""
    parts, last_ms = [], None
    for start, end in periods:
        d = pd.Timestamp(start)
        while d <= pd.Timestamp(end):
            nxt = min(d + pd.Timedelta(days=step_days), pd.Timestamp(end) + pd.Timedelta(days=1))
            ms = fit(df[df.date < d])
            chunk = trainable(df[(df.date >= d) & (df.date < nxt)])
            if len(chunk):
                parts.append(predict(ms, chunk))
            last_ms, d = ms, nxt
    return pd.concat(parts), last_ms


def backtest(df):
    t = time.time()
    periods = [("2026-01-01", "2026-04-12"), ("2026-10-27", "2026-11-22")]
    test, ms = walk_forward(df, periods)
    print(f"backtest walk-forward {time.time()-t:.0f}s, {len(test)} lignes testées")
    test["base_ewm"] = test.ttfl_ewm.fillna(test.prev_ttfl).fillna(10)
    test["base_l10"] = test.ttfl_l10.fillna(test.prev_ttfl).fillna(10)
    for label, sub in [("2025-26 (janv-avr)", test[test.season == 2026]), ("2026-27 (en cours)", test[test.season == 2027])]:
        res = {}
        for c in ["base_l10", "base_ewm", "proj"]:
            mae = (sub[c] - sub.ttfl).abs().mean()
            top = sub[sub.proj >= 30]
            tot, _ = simulate_season(sub, c)
            res[c] = dict(erreur_moy=round(mae, 2), erreur_stars=round((top[c] - top.ttfl).abs().mean(), 2),
                          total_saison=round(tot))
        oracle, _ = simulate_season(sub, "ttfl")
        cover = ((sub.ttfl >= sub.p10) & (sub.ttfl <= sub.p90)).mean()
        print(f"\n=== {label} : {sub.date.nunique()} nuits ===")
        print(pd.DataFrame(res).T)
        print(f"oracle : {oracle:.0f} | couverture intervalle 10-90 : {cover:.0%}")
    imp = permutation_like(ms, test[test.season == 2027])
    print("variables les plus influentes :", imp)
    return ms, test


def permutation_like(ms, test, n=12):
    from sklearn.inspection import permutation_importance
    s = test.sample(min(4000, len(test)), random_state=0)
    r = permutation_importance(ms["mean"], s[ms["cols"]], s.ttfl - s.base, n_repeats=3, random_state=0,
                               scoring="neg_mean_absolute_error")
    imp = pd.Series(r.importances_mean, index=ms["cols"])  # importance sur l'ajustement.sort_values(ascending=False)
    return imp.head(n).round(2).to_dict()


if __name__ == "__main__":
    t = time.time()
    import sys
    if "--cached" in sys.argv:
        df = pd.read_parquet(ROOT / "data" / "features.parquet")
    else:
        box, games = F.load(ROOT)
        df = F.build(box, games)
        print(f"features {time.time()-t:.0f}s, {len(df)} lignes")
        df.to_parquet(ROOT / "data" / "features.parquet")
    if "--no-backtest" not in sys.argv:
        backtest(df)
    ms = fit(df)  # modèle final sur tout l'historique
    import availability as A
    box, games = F.load(ROOT) if "--cached" in sys.argv else (box, games)
    ms["avail"] = A.fit(A.panel(box, games))
    pickle.dump(ms, open(MODELS, "wb"))
    print("modèle final sauvegardé")
