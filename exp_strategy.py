import pandas as pd, model as M, features as F, strategy as S, warnings
warnings.filterwarnings("ignore")
df = pd.read_parquet(M.ROOT/"data"/"features.parquet")
games = pd.read_parquet(M.ROOT/"data"/"games.parquet")
test, _ = M.walk_forward(df, [("2025-12-01","2026-04-12"),("2026-10-27","2026-11-22")])
# historique complet (dnp inclus) pour l'état des joueurs + proj sur les lignes testées
hist = df.drop_duplicates(["game_id","player_id"]).merge(test[["game_id","player_id","proj"]].drop_duplicates(["game_id","player_id"]), on=["game_id","player_id"], how="left")
pred = hist[hist.proj.notna()].copy()
full = hist.copy()
def run(start,end,h,dec):
    # state calculé sur l'historique complet ; tonight = lignes jouées avec proj
    S_player_state = S.player_state
    tot_o, tot_g, log = 0,0,None
    o,g,log = S.backtest_full(full, pred, games, start, end, h, dec)
    return o,g,log
for label,(a,b) in {"2025-26 janv-avr":("2026-01-01","2026-04-12"),"2026-27":("2026-11-01","2026-11-22")}.items():
    for h,dec in [(30,1.0),(30,0.98),(14,1.0)]:
        o,g,log = run(pd.Timestamp(a),pd.Timestamp(b),h,dec)
        print(f"{label} horizon={h} decay={dec}: optimisé {o:.0f} vs glouton {g:.0f} ({(o-g)/len(log):+.2f}/soir, {len(log)} soirs)")
log.to_csv("data/strategy_log.csv", index=False)
