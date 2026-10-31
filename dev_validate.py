"""Development walk-forward validation, kept separate from the official final check.

    python dev_validate.py [model ...]

Predicts each month of DEV_MONTHS from models trained only on sales that closed at least
`horizon` months before that month starts. The real test (Jun-Sep 2026, trained on sales up
to May 2026) has a gap of 0-3 whole months, so every month is scored at horizons 0-3 and the
average over horizons is the number that should track the hidden-test error.

Use this for feature selection and tuning. Only run `python run.py validate` (Dec 2025 - May
2026) a few times, at the end.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data
import features
import models

ROOT = Path(__file__).parent
DEV_MONTHS = pd.period_range("2025-06", "2025-11", freq="M")
# FINAL_CHECK=1 scores the official validation months instead, at the same horizons 0-3.
# Run it once, after the models are frozen; never use it to choose anything.
FINAL_CHECK = os.environ.get("FINAL_CHECK") == "1"
if FINAL_CHECK:
    DEV_MONTHS = pd.period_range("2025-12", "2026-05", freq="M")
HORIZONS = [0, 1, 2, 3]
# DEV_TRACKS=off_market python dev_validate.py ...  scores one track only (halves the run time).
TRACKS = os.environ.get("DEV_TRACKS", "off_market,list_aware").split(",")


class PassLineBaseline:
    """The "maximum error" bar: median sale of the same type in the same community over the
    12 months before the home was listed. Falls back to the same type city-wide."""

    def fit(self, train):
        self.sales = train[["community", "house_type_name", "date_sold", "sold_price"]]
        self.by_group = dict(list(self.sales.groupby(["community", "house_type_name"])))
        self.by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    @staticmethod
    def window_median(sales, listed):
        recent = sales[(sales["date_sold"] < listed) & (sales["date_sold"] >= listed - pd.DateOffset(months=12))]
        return float(recent["sold_price"].median()) if len(recent) else np.nan

    def predict(self, homes):
        out = []
        for _, home in homes.iterrows():
            group = self.by_group.get((home["community"], home["house_type_name"]))
            value = np.nan if group is None else self.window_median(group, home["date_listed"])
            if np.isnan(value):
                value = self.window_median(self.by_type[home["house_type_name"]], home["date_listed"])
            out.append(value)
        return np.array(out)


class TrimOutliers:
    """Wrap a model: drop training sales outside the 0.1-99.9 percentile of their property
    type's price before fitting. Prediction is unchanged (every test home still gets a price)."""

    def __init__(self, model):
        self.model = model

    def fit(self, train):
        by_type = train.groupby("house_type_name")["sold_price"]
        low, high = by_type.transform(lambda s: s.quantile(0.001)), by_type.transform(lambda s: s.quantile(0.999))
        self.model.fit(train[(train["sold_price"] >= low) & (train["sold_price"] <= high)])
        return self

    def predict(self, homes):
        return self.model.predict(homes)


def feature_model(track):
    return models.FeatureModel(
        features.list_aware_features if track == "list_aware" else features.off_market_features)


def off_market_variant(drop=(), half_life_days=None, max_staleness_days=features.MAX_STALENESS_DAYS,
                       bias_correction_days=None, **params):
    """A FeatureModel for off-market with some features dropped or other settings, for ablations."""
    def make_features(homes, history, training=False):
        X = features.off_market_features(homes, history, training, max_staleness_days)
        return X.drop(columns=list(drop))
    return lambda track: models.FeatureModel(make_features, half_life_days, bias_correction_days, **params)


def list_aware_variant(sale_to_list=True, versus_market=True, max_staleness_days=features.MAX_STALENESS_DAYS,
                       relative_to_list=True, half_life_days=None, **params):
    """A FeatureModel for list-aware with some feature groups or settings changed, for ablations."""
    def make_features(homes, history, training=False):
        return features.list_aware_features(homes, history, training, sale_to_list, versus_market,
                                            max_staleness_days)
    return lambda track: models.FeatureModel(make_features, half_life_days, None, relative_to_list, **params)


MODELS = {
    "passline": lambda track: PassLineBaseline(),
    "starter": models.make_baseline,
    "final": models.make_model,
    "feature": feature_model,
    "feature_trim": lambda track: TrimOutliers(feature_model(track)),
    # Phase 1 step 5 (trend) ablations, off-market only.
    "anchor_only": off_market_variant(max_staleness_days=0),
    "stale120": off_market_variant(),
    "stale120_hl730": off_market_variant(half_life_days=730),
    "stale120_hl365": off_market_variant(half_life_days=365),
    "stale120_nomonths": off_market_variant(drop=["months"]),
    "stale120_hl180": off_market_variant(half_life_days=180),
    "stale120_hl365_bc": off_market_variant(half_life_days=365, bias_correction_days=120),
    # 2x2 check of staleness vs recency weights (completes anchor_only / stale120 / stale120_hl365).
    "anchor_hl365": off_market_variant(half_life_days=365, max_staleness_days=0),
    # The final off-market model with staleness training switched off.
    "final_nostale": off_market_variant(half_life_days=365, max_staleness_days=0, **models.OFF_MARKET_PARAMS),
    # The final list-aware model without sklearn's early stopping (which holds out a random 10%).
    "list_final_noes": list_aware_variant(half_life_days=365, early_stopping=False),
    # Light tuning of the trees on top of #12 (no early stopping: it holds out a random 10%).
    "tune_a": off_market_variant(half_life_days=365, max_iter=300, learning_rate=0.05, early_stopping=False),
    "tune_b": off_market_variant(half_life_days=365, max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                                 min_samples_leaf=50, early_stopping=False),
    # Phase 2 (list-aware) steps.
    "list_price_target": list_aware_variant(False, False, 0, relative_to_list=False),
    "list_ratio": list_aware_variant(False, False, 0),
    "list_ratio_s2l": list_aware_variant(True, False),
    "list_ratio_s2l_vm": list_aware_variant(True, True),
    "list_hl365": list_aware_variant(half_life_days=365),
    "list_hl365_a": list_aware_variant(half_life_days=365, max_iter=300, learning_rate=0.05, early_stopping=False),
    "list_hl365_b": list_aware_variant(half_life_days=365, max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                                       min_samples_leaf=50, early_stopping=False),
    "list_hl365_tuned": list_aware_variant(half_life_days=365, max_iter=1500, learning_rate=0.03,
                                           max_leaf_nodes=127, min_samples_leaf=50, early_stopping=False),
    "tune_c": off_market_variant(half_life_days=365, max_iter=1500, learning_rate=0.03, max_leaf_nodes=127,
                                 min_samples_leaf=50, early_stopping=False),
}


def ape(predicted, sold):
    return 100 * np.abs(predicted - sold) / sold


def evaluate(name, listings):
    rows = []
    for month in DEV_MONTHS:
        homes = listings[listings["date_sold"].dt.to_period("M") == month]
        for horizon in HORIZONS:
            cutoff = (month - horizon).start_time
            train = listings[listings["date_sold"] < cutoff]
            for track in TRACKS:
                predicted = MODELS[name](track).fit(train).predict(data.hide_answers(homes, track))
                rows.append(pd.DataFrame({
                    "model": name, "track": track, "month": str(month), "horizon": horizon,
                    "ml_num": homes["ml_num"], "house_type_name": homes["house_type_name"],
                    "sold_price": homes["sold_price"], "predicted_price": predicted}))
        print(f"  {name} {month} done", flush=True)
    return pd.concat(rows)


def summarize(results):
    results = results.assign(ape=ape(results["predicted_price"], results["sold_price"]),
                             signed=100 * (results["predicted_price"] / results["sold_price"] - 1))
    by_h = results.pivot_table(index=["model", "track"], columns="horizon", values="ape", aggfunc="median")
    by_h.columns = [f"MdAPE h={h}" for h in by_h.columns]
    overall = results.groupby(["model", "track"]).agg(
        MdAPE=("ape", "median"), MAPE=("ape", "mean"), median_signed=("signed", "median"))
    signed_h = results.pivot_table(index=["model", "track"], columns="horizon", values="signed", aggfunc="median")
    signed_h.columns = [f"signed h={h}" for h in signed_h.columns]
    return overall.join(by_h).join(signed_h).round(2)


def signed_by_month(results):
    """Median signed error (%) per predicted month: above 0 means the model prices too high."""
    signed = 100 * (results["predicted_price"] / results["sold_price"] - 1)
    return results.assign(signed=signed).pivot_table(
        index=["model", "track"], columns="month", values="signed", aggfunc="median").round(2)


if __name__ == "__main__":
    names = sys.argv[1:] or list(MODELS)
    listings = data.load("listings")
    results = pd.concat([evaluate(name, listings) for name in names])
    (ROOT / "outputs").mkdir(exist_ok=True)
    prefix = "final_check" if FINAL_CHECK else "dev_predictions"
    results.to_csv(ROOT / "outputs" / f"{prefix}_{'_'.join(names)}.csv", index=False)
    print(summarize(results).to_string())
    print("\nMedian signed error % by month:")
    print(signed_by_month(results).to_string())
