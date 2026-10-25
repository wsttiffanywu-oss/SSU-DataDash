"""Prediction intervals from the model's own past errors (split conformal).

dev_validate.py predicts homes whose sold prices we know, at the same 0-3 month gaps as the
real test. Each of those predictions has an error in log terms, log(sold / predicted). For a
new home, the interval is its predicted price times the 10th and 90th percentile of the errors
of similar homes (same property type and price band): if future errors look like past ones,
about 80% of homes sell inside it. The errors are not centred on zero (the models still price
a little high), so the intervals are not symmetric around the prediction, which is intended.
"""
import numpy as np
import pandas as pd

LEVELS = (0.8, 0.9)
MIN_GROUP = 200   # groups with fewer calibration homes fall back to the property type, then the track


class ConformalIntervals:
    """fit() on dev predictions of one track (house_type_name, predicted_price, sold_price);
    predict() gives lower/upper bounds for each level for new homes."""

    def __init__(self, levels=LEVELS, n_bands=3):
        self.levels = levels
        self.n_bands = n_bands   # price bands within each property type, by predicted price; 1 = none

    def band(self, kinds, predicted):
        """Price band of each home: 0 (cheapest third of its type) to n_bands - 1."""
        out = np.zeros(len(predicted), dtype=int)
        for kind, edges in self.edges.items():
            mask = (kinds == kind).to_numpy()
            out[mask] = np.searchsorted(edges, predicted[mask])
        return out

    def quantiles(self, log_error):
        return {level: np.quantile(log_error, [(1 - level) / 2, (1 + level) / 2]) for level in self.levels}

    def fit(self, results):
        kinds = results["house_type_name"]
        predicted = results["predicted_price"].to_numpy(dtype=float)
        log_error = np.log(results["sold_price"].to_numpy() / predicted)
        cuts = np.linspace(0, 1, self.n_bands + 1)[1:-1]
        self.edges = {kind: np.quantile(predicted[(kinds == kind).to_numpy()], cuts) for kind in kinds.unique()}
        bands = self.band(kinds, predicted)
        frame = pd.DataFrame({"kind": kinds.to_numpy(), "band": bands, "log_error": log_error})
        self.overall = self.quantiles(log_error)
        self.by_kind = {k: self.quantiles(g["log_error"]) for k, g in frame.groupby("kind") if len(g) >= MIN_GROUP}
        self.by_band = {k: self.quantiles(g["log_error"]) for k, g in frame.groupby(["kind", "band"])
                        if len(g) >= MIN_GROUP}
        return self

    def predict(self, kinds, predicted):
        """Lower and upper bounds (in dollars) for each level, one row per home."""
        kinds = pd.Series(kinds).reset_index(drop=True)
        predicted = np.asarray(predicted, dtype=float)
        bands = self.band(kinds, predicted)
        out = {}
        for level in self.levels:
            low, high = np.empty(len(predicted)), np.empty(len(predicted))
            for i, (kind, band) in enumerate(zip(kinds, bands)):
                q = self.by_band.get((kind, band), self.by_kind.get(kind, self.overall))[level]
                low[i], high[i] = q
            pct = int(round(100 * level))
            out[f"lower_{pct}"] = predicted * np.exp(low)
            out[f"upper_{pct}"] = predicted * np.exp(high)
        return pd.DataFrame(out)


def coverage(bounds, sold, predicted, level):
    """Share of homes whose sold price falls inside the interval, and the interval's median
    width in % of the predicted price (what a user would see)."""
    pct = int(round(100 * level))
    low, high = bounds[f"lower_{pct}"].to_numpy(), bounds[f"upper_{pct}"].to_numpy()
    sold = np.asarray(sold, dtype=float)
    inside = (sold >= low) & (sold <= high)
    return float(inside.mean()), float(np.median(100 * (high - low) / np.asarray(predicted, dtype=float)))
