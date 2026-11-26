"""
compare_models.py - Phase 4: a fair race between three models.
Same data, same cleaning, same features, same four practice exams, default-style settings.
Results are saved to results/results.csv next to the baselines.
"""
import pandas as pd
from sklearn.linear_model import Ridge                  # linear regression with a 'leash'
from sklearn.preprocessing import StandardScaler        # puts features on the same ruler
from sklearn.pipeline import make_pipeline              # chains steps into an assembly line
from sklearn.ensemble import RandomForestRegressor      # 200 trees voting
from lightgbm import LGBMRegressor                      # trees that fix each other's mistakes

from common import load_data, evaluate_model            # our shared toolbox
from runner import make_runner                          # turns a model into learn-then-predict

SEED = 42                                               # fixed dice, so results repeat
train, _ = load_data()                                  # we only need the labelled history (_ = ignore the second table)

# Each entry is: model name -> a recipe that builds a fresh, untrained model.
candidates = {
    "linear_ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200,        # number of trees in the crowd
        min_samples_leaf=10,     # each leaf needs 10+ loads (stops memorising single loads)
        n_jobs=-1,               # use every CPU core to go faster
        random_state=SEED),
    "lightgbm": lambda: LGBMRegressor(
        n_estimators=300,        # number of trees built one after another
        learning_rate=0.05,      # how big a correction each tree may make
        num_leaves=31,           # size limit per tree
        random_state=SEED,
        verbose=-1),             # keep the output quiet
}

print("RACE: each model predicts 4 months it has not seen")
for name, build in candidates.items():
    evaluate_model(name, make_runner(build), train)     # runs the 4 exams and saves the average

# ---- The scoreboard ------------------------------------------------------------
results = pd.read_csv("results/results.csv").sort_values("mae").reset_index(drop=True)
# Best no-learning guess = the bar every model must clear.
best_baseline_mae = results.loc[results["model"].str.startswith("baseline"), "mae"].min()
results["mae_better_than_baseline_%"] = ((best_baseline_mae - results["mae"])
                                         / best_baseline_mae * 100).round(1)
print("\nSCOREBOARD (average over 4 tests, best MAE first)")
print(results.round(2).to_string(index=False))