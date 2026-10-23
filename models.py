"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import features


# Off-market tree settings, chosen on the dev months 2025-06 to 2025-11 (experiments.md #19).
OFF_MARKET_PARAMS = {"max_iter": 1500, "learning_rate": 0.03, "max_leaf_nodes": 127,
                     "min_samples_leaf": 50, "early_stopping": False}


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        # Phase 1 placeholder: property features + list price, default trees (experiments.md #9).
        return FeatureModel(features.list_aware_features)
    # Market features with simulated staleness, recent sales weighted more (experiments.md #12, #19).
    return FeatureModel(features.off_market_features, half_life_days=365, **OFF_MARKET_PARAMS)


def make_baseline(track):
    """The starter's baselines, kept for comparison."""
    return ListPriceBaseline() if track == "list_aware" else NearestSalesBaseline(k=10)


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features, half_life_days=None, bias_correction_days=None, **params):
        """half_life_days: if set, a sale that many days older than the newest training sale
        counts half as much in training. bias_correction_days: if set, see fit_bias().
        params go to HistGradientBoostingRegressor."""
        self.make_features = make_features
        self.half_life_days = half_life_days
        self.bias_correction_days = bias_correction_days
        self.params = params
        self.log_bias = 0.0
        self.model = HistGradientBoostingRegressor(
            **{"random_state": 0, "categorical_features": "from_dtype", **params})

    def encode(self, X):
        """Text columns become categories with the vocabulary seen in fit(), so the same
        value gets the same code in training and prediction. Unseen values become missing."""
        X = X.copy()
        for column, categories in self.categories.items():
            X[column] = pd.Categorical(X[column], categories=categories)
        return X

    def fit_bias(self, train):
        """Measure the model's systematic error the way the test will see it, using training
        sales only: train a copy on all but the last bias_correction_days, predict those last
        days (their market data is 0 to bias_correction_days old, like the test homes'), and keep
        the median log error, which predict() then removes."""
        split = train["date_sold"].max() - pd.Timedelta(days=self.bias_correction_days)
        early, late = train[train["date_sold"] <= split], train[train["date_sold"] > split]
        late = late[late["date_listed"] > split]   # listed after the copy's data ends, like the test
        copy = FeatureModel(self.make_features, self.half_life_days, **self.params).fit(early)
        self.log_bias = float(np.median(np.log(copy.predict(late)) - np.log(late["sold_price"])))

    def fit(self, train):
        if self.bias_correction_days:
            self.fit_bias(train)
        # The training sales are also the history that market features look back on.
        self.history = train
        X = self.make_features(train, self.history, training=True)
        self.categories = {c: sorted(X[c].dropna().unique()) for c in X.columns if X[c].dtype == object}
        weight = None
        if self.half_life_days:
            age = (train["date_sold"].max() - train["date_sold"]).dt.days
            weight = 0.5 ** (age / self.half_life_days)
        self.model.fit(self.encode(X), np.log(train["sold_price"]), sample_weight=weight)
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.encode(self.make_features(homes, self.history))) - self.log_bias)
