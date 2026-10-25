"""EDIT ME: turn a table of homes into numbers a model can learn from.

Each function gets a table of homes with the columns of its test file (so never
sold_price or date_sold) and returns a table of numbers, one row per home.
FeatureModel in models.py uses them; switch it on in make_model() to try your features.

Ideas to try:
- lot size: lot_front * lot_depth (empty for condos, which is fine for tree models)
- garage, house_style, municipality
- how pricey a community is: the median price per square foot of its past sales
  (work it out from the training sales only, never from the homes you predict)
- how prices have moved recently in the area or for that property type
- words in the remarks, such as "renovated" or "needs work"
- off-market and list-aware can use different features: list-aware has list_price
"""
import numpy as np
import pandas as pd

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def past_window_mean(homes, history, keys, days, value="log_ppsf"):
    """For each home: the mean `value` (a column of history) of the history sales in its `keys` group
    that closed in the `days` days before its `as_of` date, and how many there were.

    `as_of` is never after the listing date, and a sale on the `as_of` day itself is not used.
    The same rule applies to training rows and to the homes being predicted, so a training home
    never sees its own sale or anything after it."""
    mean = pd.Series(np.nan, index=homes.index)
    count = pd.Series(0, index=homes.index)
    past_groups = dict(list(history.groupby(keys)))
    for key, rows in homes.groupby(keys):
        past = past_groups.get(key)
        if past is None:
            continue
        past = past.sort_values("date_sold")
        dates = past["date_sold"].to_numpy()
        cumulative = np.concatenate([[0.0], np.cumsum(past[value].to_numpy())])
        as_of = rows["as_of"].to_numpy()
        end = np.searchsorted(dates, as_of, side="left")
        start = np.searchsorted(dates, as_of - np.timedelta64(days, "D"), side="left")
        n = end - start
        with np.errstate(invalid="ignore", divide="ignore"):
            mean[rows.index] = np.where(n > 0, (cumulative[end] - cumulative[start]) / n, np.nan)
        count[rows.index] = n
    return mean, count


def as_of_dates(homes, history, training=False, max_staleness_days=0):
    """The date each home's market windows end (see market_features). Deterministic, so every
    feature group of one training table gets the same random staleness."""
    if training:
        rng = np.random.default_rng(0)
        lag = rng.integers(0, max_staleness_days + 1, len(homes)) if max_staleness_days else 0
        return homes["date_listed"] - pd.to_timedelta(lag, unit="D")
    return homes["date_listed"].clip(upper=history["date_sold"].max() + pd.Timedelta(days=1))


def market_features(homes, history, training=False, max_staleness_days=0):
    """How expensive the market around a home was just before it was listed, from past sales
    only: mean log(price per sq ft) by community x type, community, and type city-wide.

    The windows end at `as_of`. For a home to predict that is the listing date, or the day after
    the last known sale if it was listed later (so the windows hold the latest data, not empty
    months). Training homes always have data up to their listing date, unlike the test homes,
    which are listed 0-4 months after the data ends. With max_staleness_days > 0, each training
    home instead gets a random as_of 0 to max_staleness_days before its listing, and
    `staleness_days` tells the model how old its market data is, so it can learn from 2021-2025
    how far prices move while the data is that old."""
    history = history.assign(
        log_ppsf=np.log(history["sold_price"] / square_feet(history["house_area"])), all="all")
    homes = homes.assign(all="all", as_of=as_of_dates(homes, history, training, max_staleness_days))
    X = pd.DataFrame(index=homes.index)
    if max_staleness_days:
        X["staleness_days"] = (homes["date_listed"] - homes["as_of"]).dt.days
    groups = {"ct": ["community", "house_type_name"], "c": ["community"],
              "t": ["house_type_name"], "city": ["all"]}
    windows = {"ct": [91, 182, 365], "c": [91, 365], "t": [30, 91, 365], "city": [30, 91, 365]}
    for name, keys in groups.items():
        for days in windows[name]:
            X[f"{name}_ppsf_{days}d"], n = past_window_mean(homes, history, keys, days)
            if days == 365:
                X[f"{name}_n_365d"] = n
    # Recent momentum (last 3 months vs last 12) and the location premium over the type average.
    X["t_momentum"] = X["t_ppsf_91d"] - X["t_ppsf_365d"]
    X["city_momentum"] = X["city_ppsf_91d"] - X["city_ppsf_365d"]
    X["ct_premium"] = X["ct_ppsf_365d"] - X["t_ppsf_365d"]
    # A naive estimate: the neighbourhood's recent price per sq ft times this home's size.
    X["ct_naive_log_price"] = X["ct_ppsf_182d"] + np.log(square_feet(homes["house_area"]))
    return X


def sale_to_list_features(homes, history, training=False, max_staleness_days=0):
    """How far below or above their list price homes closed just before this one was listed:
    mean log(sold / list) of past sales by community x type, community, type and city, using
    the same windows ending at `as_of` as market_features()."""
    history = history.assign(log_sale_to_list=np.log(history["sold_price"] / history["list_price"]), all="all")
    homes = homes.assign(all="all", as_of=as_of_dates(homes, history, training, max_staleness_days))
    X = pd.DataFrame(index=homes.index)
    groups = {"ct": ["community", "house_type_name"], "c": ["community"],
              "t": ["house_type_name"], "city": ["all"]}
    windows = {"ct": [91, 365], "c": [365], "t": [30, 91, 365], "city": [30, 91, 365]}
    for name, keys in groups.items():
        for days in windows[name]:
            X[f"{name}_s2l_{days}d"], n = past_window_mean(homes, history, keys, days, "log_sale_to_list")
            if days == 365:
                X[f"{name}_s2l_n_365d"] = n
    X["city_s2l_momentum"] = X["city_s2l_91d"] - X["city_s2l_365d"]
    return X


def list_vs_market(homes, market):
    """How the list price compares with what recent sales nearby suggest (log ratios; above 0:
    listed above the neighbourhood's recent price per sq ft times this home's size)."""
    log_list = np.log(homes["list_price"])
    log_sqft = np.log(square_feet(homes["house_area"]))
    X = pd.DataFrame(index=homes.index)
    X["list_vs_ct_182d"] = log_list - market["ct_naive_log_price"]
    X["list_vs_ct_365d"] = log_list - (market["ct_ppsf_365d"] + log_sqft)
    X["list_vs_t_91d"] = log_list - (market["t_ppsf_91d"] + log_sqft)
    X["list_vs_c_365d"] = log_list - (market["c_ppsf_365d"] + log_sqft)
    return X


# Phrases in the remarks, grouped by what they say about the home. Picked by reading which
# words go with prices above or below the neighbourhood average in sales before June 2025
# (before the dev months), then grouped by hand.
REMARK_PATTERNS = {
    "rm_renovated": r"fully (?:redone|renovated|updated)|completely (?:redone|renovated)|beautifully renovated"
                    r"|renovated|turn-?key|move-in ready|gut(?:ted)? renovat",
    "rm_luxury_finish": r"quartz|heated (?:bathroom )?floors|designer light|smooth ceilings|built-in appliances"
                        r"|custom kitchen|frameless glass|spa-like|pot lights|wide-plank",
    "rm_new_systems": r"new roof|high[- ]efficiency furnace|new central air|new windows|updated (?:electrical|plumbing)",
    "rm_finished_basement": r"finished basement|renovated basement|rec room",
    "rm_landscaped": r"landscaped",
    "rm_dated": r"original (?:finishes|kitchen|condition|bathroom|cabinet)|\bdated\b|\bworn\b|\btired\b|older appliances"
                r"|needs? (?:updating|work|tlc|some love)|personal touch|bring your (?:ideas|vision|imagination)",
    "rm_major_repairs": r"needs? (?:significant|major|a new)|need replacing|waterproofing|near (?:the )?end of (?:its |their )?life"
                        r"|older furnace|contractor|\brenovate\b|as-is|overgrown|handyman|structural|investor",
    "rm_motivated": r"motivated|bring your offers?",
}


def remark_features(homes):
    """0/1 columns for each group of phrases in REMARK_PATTERNS, plus a net condition score."""
    text = homes["remarks"].fillna("").str.lower()
    X = pd.DataFrame(index=homes.index)
    for name, pattern in REMARK_PATTERNS.items():
        X[name] = text.str.contains(pattern, regex=True).astype(int)
    X["rm_condition_score"] = (X["rm_renovated"] + X["rm_luxury_finish"] + X["rm_new_systems"]
                               - X["rm_dated"] - X["rm_major_repairs"])
    X["rm_length"] = text.str.len()
    return X


def property_features(homes):
    """What is known about the home itself and when it was listed."""
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(homes["house_area"])
    # Ranges ("900-1099") are a coarser report than an exact figure; let the model tell them apart.
    X["area_is_range"] = homes["house_area"].astype(str).str.contains("-").astype(int)
    # Lots exist only for freehold types; condos stay empty, which tree models handle.
    X["lot_front"] = homes["lot_front"]
    X["lot_depth"] = homes["lot_depth"]
    X["lot_area"] = homes["lot_front"] * homes["lot_depth"]
    X["garage"] = homes["garage"]
    # Text columns: FeatureModel turns them into categories learned from the training sales.
    X["house_style"] = homes["house_style"]
    X["municipality"] = homes["municipality"]
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    # Months since January 2021, so the model can follow prices over time.
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    # Calendar month of listing (1-12), for the spring peak and the summer/autumn dip. Every
    # month appears in each training year, so trees can learn it, unlike the trend above.
    X["list_month"] = homes["date_listed"].dt.month
    # One 0/1 column per property type.
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    return X.join(remark_features(homes))


MAX_STALENESS_DAYS = 120   # the test homes are listed up to ~4 months after the data ends


def off_market_features(homes, history, training=False, max_staleness_days=MAX_STALENESS_DAYS):
    """Features for homes with no list price. `history` is the training sales, used only
    through market_features(), which looks at sales before each home's listing date.
    `training` is True when `homes` are the training sales themselves."""
    return property_features(homes).join(market_features(homes, history, training, max_staleness_days))


def list_aware_features(homes, history, training=False, sale_to_list=True, versus_market=True,
                        max_staleness_days=MAX_STALENESS_DAYS):
    """Features for homes that have a list price: the property features, the list price and,
    optionally, recent sale-to-list ratios and how the list price compares with the market.
    (Without these, the off-market market features made list-aware worse: experiments.md #7.)"""
    X = property_features(homes)
    X["list_price"] = homes["list_price"]
    if sale_to_list:
        X = X.join(sale_to_list_features(homes, history, training, max_staleness_days))
    if versus_market:
        market = market_features(homes, history, training, max_staleness_days)
        X = X.join(list_vs_market(homes, market))
        X["log_list_per_sqft"] = np.log(homes["list_price"] / square_feet(homes["house_area"]))
        if max_staleness_days and "staleness_days" not in X:
            X["staleness_days"] = market["staleness_days"]
    elif sale_to_list and max_staleness_days:
        X["staleness_days"] = (homes["date_listed"] - as_of_dates(homes, history, training, max_staleness_days)).dt.days
    return X
