"""Leakage checks for the market features: a home may only see sales that closed before it was listed."""
import numpy as np
import pandas as pd
import pytest

import features
from test_starter import make_listings


@pytest.mark.parametrize("make", [features.market_features, features.sale_to_list_features])
@pytest.mark.parametrize("options", [{}, {"training": True, "max_staleness_days": 120}])
def test_market_features_ignore_later_and_own_sales(make, options):
    listings = make_listings(500)
    before = make(listings, listings, **options)
    # Make every sale wildly expensive: only homes listed after that sale may change.
    changed = listings.assign(sold_price=listings["sold_price"] * 100)
    some_later_home_changed = False
    for i in [0, 10, 200, 300, 400]:
        tweaked = listings.copy()
        tweaked.loc[i, "sold_price"] = changed.loc[i, "sold_price"]
        after = make(tweaked, tweaked, **options)
        # The home's own sale never feeds its own features.
        pd.testing.assert_series_equal(before.loc[i], after.loc[i])
        # Homes listed on or before that sale closed are unaffected.
        unaffected = listings["date_listed"] <= listings.loc[i, "date_sold"]
        pd.testing.assert_frame_equal(before[unaffected], after[unaffected])
        some_later_home_changed |= not before[~unaffected].equals(after[~unaffected])
    # ...and the check is not vacuous: later homes do see the changes.
    assert some_later_home_changed


def test_market_features_match_a_direct_calculation():
    listings = make_listings(500)
    X = features.market_features(listings, listings)
    log_ppsf = np.log(listings["sold_price"] / features.square_feet(listings["house_area"]))
    for i in [3, 77, 401]:
        home = listings.loc[i]
        window = ((listings["house_type_name"] == home["house_type_name"])
                  & (listings["date_sold"] < home["date_listed"])
                  & (listings["date_sold"] >= home["date_listed"] - pd.Timedelta(days=91)))
        expected = log_ppsf[window].mean() if window.any() else np.nan
        assert np.isclose(X.loc[i, "t_ppsf_91d"], expected, equal_nan=True)


def test_homes_listed_after_the_data_ends_use_the_latest_data():
    history = make_listings(500)
    last_sale = history["date_sold"].max()
    home = history.iloc[[0]].assign(date_listed=last_sale + pd.Timedelta(days=60))
    X = features.market_features(home, history, max_staleness_days=120)
    assert X["staleness_days"].iloc[0] == 59
    assert X["city_n_365d"].iloc[0] == (history["date_sold"] > last_sale - pd.Timedelta(days=364)).sum()
