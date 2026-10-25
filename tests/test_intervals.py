"""Checks for the conformal intervals on made-up errors."""
import numpy as np
import pandas as pd

from intervals import ConformalIntervals, coverage


def made_up_results(n, seed, spread):
    rng = np.random.default_rng(seed)
    kinds = rng.choice(["Condo Apt", "Detached"], n)
    predicted = rng.uniform(4e5, 2e6, n)
    noise = np.where(kinds == "Detached", spread * 2, spread)
    sold = predicted * np.exp(rng.normal(0.02, noise))
    return pd.DataFrame({"house_type_name": kinds, "predicted_price": predicted, "sold_price": sold})


def test_coverage_matches_the_level_on_new_homes_from_the_same_errors():
    model = ConformalIntervals().fit(made_up_results(20000, 0, 0.1))
    new = made_up_results(20000, 1, 0.1)
    bounds = model.predict(new["house_type_name"], new["predicted_price"])
    for level in (0.8, 0.9):
        covered, _ = coverage(bounds, new["sold_price"], new["predicted_price"], level)
        assert abs(covered - level) < 0.02


def test_noisier_types_get_wider_intervals():
    model = ConformalIntervals().fit(made_up_results(20000, 0, 0.1))
    bounds = model.predict(pd.Series(["Condo Apt", "Detached"]), [1e6, 1e6])
    widths = bounds["upper_80"] - bounds["lower_80"]
    assert widths[1] > 1.5 * widths[0]


def test_unseen_or_small_groups_fall_back_to_the_whole_track():
    results = made_up_results(5000, 0, 0.1)
    results.loc[:9, "house_type_name"] = "Co-Op Apt"   # only 10 homes: too few for their own interval
    model = ConformalIntervals().fit(results)
    bounds = model.predict(pd.Series(["Co-Op Apt", "Semi-Detached"]), [1e6, 1e6])
    overall = 1e6 * np.exp(model.overall[0.8])
    assert np.allclose(bounds.loc[0, ["lower_80", "upper_80"]], overall)
    assert np.allclose(bounds.loc[1, ["lower_80", "upper_80"]], overall)
