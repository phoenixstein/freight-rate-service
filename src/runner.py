"""
runner.py - turns any scikit-learn-style model into the 'learn, then predict' function
that evaluate_model() expects. Every model gets the SAME cleaning, features and target,
so the race between models is fair.
"""
import numpy as np
from common import learn_fill_values, clean_inputs, make_features   # our shared toolbox


def make_runner(build_model):
    """build_model: a function that returns a fresh, untrained model each time it is called."""

    def fit_predict(fit_df, test_df):
        # 1. Typical fill-in values come ONLY from the rows we learn from (no peeking at the test month).
        fills = learn_fill_values(fit_df)
        # 2. Clean the learning rows and turn them into model inputs (X).
        X_fit = make_features(clean_inputs(fit_df, fills))
        # 3. The answer we teach the model: log of price PER MILE (comparable for short and long trips).
        y_fit = np.log(fit_df["posted_rate"] / fit_df["distance"])
        # 4. Build a fresh model and let it LEARN the pattern from the history.
        model = build_model()
        model.fit(X_fit, y_fit)
        # 5. Apply the SAME fixes to the test month, then build its inputs.
        test_clean = clean_inputs(test_df, fills)
        X_test = make_features(test_clean)
        # 6. Predict, undo the log (exp), and multiply by distance: dollars per mile x miles = dollars.
        per_mile = np.exp(model.predict(X_test))
        return per_mile * test_clean["distance"].values

    return fit_predict