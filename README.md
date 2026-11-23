# TTFL Coach

Aide au choix du pick TTFL quotidien : projection du score TTFL de chaque joueur NBA du soir,
fourchette 10-90 %, raisons de l'ajustement, blessés, et suivi de la règle des 30 jours.

## Fonctionnement

| Fichier | Rôle |
|---|---|
| `fetch.py` | Collecte ESPN (box scores, cotes, blessures), cache dans `data/raw/` |
| `features.py` | Variables sans fuite : forme, minutes, défense adverse par poste, rythme, cotes, fatigue, coéquipiers absents |
| `model.py` | Modèle « niveau de base + ajustement du soir » (gradient boosting) + backtest walk-forward |
| `availability.py` | Probabilité de jouer : modèle de présence (historique) + lecture du rapport de blessures du jour |
| `strategy.py` | Affectation optimale joueurs → soirs sous la contrainte des 30 jours (expérimental) |
| `predict.py` | Pipeline quotidien → `site/data.json` |
| `site/index.html` | Interface mobile ; les picks sont stockés sur l'appareil |

## Commandes

```bash
python fetch.py            # met à jour les matchs terminés
python model.py            # recalcule les variables, backtest, réentraîne
python predict.py          # projections du soir -> site/data.json
python -m http.server 8765 --directory site
```

## Résultats du backtest (réentraînement toutes les 2 semaines, données jamais vues)

- Erreur moyenne : 8,3 pts par joueur, contre 8,5-8,6 pour une moyenne récente.
- L'intervalle 10-90 % contient le score réel dans environ 80 % des cas : il est bien calibré.
- L'optimisation sur 30 jours n'apporte pas de gain mesurable en points attendus par rapport au
  « meilleur du soir ». L'interface signale donc seulement quand un joueur a un meilleur soir dans les 10 jours.

## Risque d'absence

Le classement se fait sur le **score espéré** = projection × probabilité de jouer.
Un joueur qui ne joue pas rapporte 0 : sur une saison simulée, ce classement réduit les picks absents
de 27 à 10 sur 121 soirs. Ce gain est surestimé, car la simulation ne voit pas le rapport de blessures.
