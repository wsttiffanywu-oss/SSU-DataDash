"""Phase 3: prediction intervals and error analysis, from the dev predictions of the final models.

    python dev_validate.py final     first: dev predictions of make_model() for both tracks
    python phase3.py intervals       coverage of conformal intervals on held-out dev months
    python phase3.py errors          where the models go wrong: tables and figures
    python phase3.py test            interval files for the test homes (after run.py predict)

Everything here reads predictions only; nothing is refitted on the final validation months.
Each dev home is predicted four times (horizons 0-3), so counts below are predictions, not homes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data
import features
from intervals import LEVELS, ConformalIntervals, coverage

ROOT = Path(__file__).parent
DEV = ROOT / "outputs" / "dev_predictions_final.csv"
TRACKS = ["off_market", "list_aware"]
# Calibrate on one half of the dev months, check on the other, then swap.
FOLDS = [(["2025-06", "2025-07", "2025-08"], ["2025-09", "2025-10", "2025-11"]),
         (["2025-09", "2025-10", "2025-11"], ["2025-06", "2025-07", "2025-08"])]
COLOURS = {"off_market": "#2a78d6", "list_aware": "#eb6834"}


def load_dev():
    if not DEV.exists():
        sys.exit(f"{DEV} does not exist yet: run `python dev_validate.py final` first")
    results = pd.read_csv(DEV)
    results["ape"] = 100 * (results["predicted_price"] - results["sold_price"]).abs() / results["sold_price"]
    results["signed"] = 100 * (results["predicted_price"] / results["sold_price"] - 1)
    return results


def summary(frame):
    return pd.Series({"predictions": len(frame), "MdAPE": frame["ape"].median(), "MAPE": frame["ape"].mean(),
                      "median signed": frame["signed"].median()})


def write_report(name, sections):
    (ROOT / "reports").mkdir(exist_ok=True)
    text = "\n\n".join(f"## {title}\n\n```\n{table.to_string()}\n```" for title, table in sections)
    (ROOT / "reports" / name).write_text(text + "\n")
    print(text)


# ---------------------------------------------------------------- intervals

def interval_coverage():
    results = load_dev()
    rows, by_group = [], []
    for track in TRACKS:
        homes = results[results["track"] == track]
        for n_bands in (1, 3):
            for calibrate, check in FOLDS:
                cal, test = homes[homes["month"].isin(calibrate)], homes[homes["month"].isin(check)]
                model = ConformalIntervals(n_bands=n_bands).fit(cal)
                bounds = model.predict(test["house_type_name"], test["predicted_price"])
                fold = f"{calibrate[0]}..{calibrate[-1]} -> {check[0]}..{check[-1]}"
                for level in LEVELS:
                    covered, width = coverage(bounds, test["sold_price"], test["predicted_price"], level)
                    rows.append({"track": track, "price bands": n_bands, "fold": fold, "level": f"{level:.0%}",
                                 "coverage": round(100 * covered, 1), "median width %": round(width, 1)})
                if n_bands == 3:
                    test = test.reset_index(drop=True).join(bounds)
                    test["inside_80"] = (test["sold_price"] >= test["lower_80"]) & (test["sold_price"] <= test["upper_80"])
                    by_group.append(test.assign(track=track, fold=fold))
    table = pd.DataFrame(rows)
    pooled = pd.concat(by_group)
    by_type = pooled.groupby(["track", "house_type_name"])["inside_80"].agg(["size", "mean"])
    by_type["mean"] = (100 * by_type["mean"]).round(1)
    by_h = pooled.pivot_table(index="track", columns="horizon", values="inside_80", aggfunc="mean").mul(100).round(1)
    by_month = pooled.pivot_table(index="track", columns="month", values="inside_80", aggfunc="mean").mul(100).round(1)
    write_report("intervals_dev.md", [
        ("Coverage on held-out dev months (calibrate on 3 months, check on the other 3)", table.set_index("track")),
        ("80% coverage by property type (3 price bands, both folds pooled)",
         by_type.rename(columns={"size": "predictions", "mean": "coverage %"})),
        ("80% coverage by horizon h", by_h), ("80% coverage by month", by_month)])


# ---------------------------------------------------------------- error analysis

def describe_homes(results):
    """Add what we know about each dev home (from listings, never the test files)."""
    listings = data.load("listings")
    remarks = features.remark_features(listings)
    homes = listings[["ml_num", "community", "house_area"]].assign(
        condition=remarks["rm_condition_score"].clip(-2, 3),
        area_is_range=listings["house_area"].astype(str).str.contains("-"),
        list_vs_sold=np.log(listings["list_price"] / listings["sold_price"]))
    out = results.merge(homes, on="ml_num", how="left")
    # Community history before the dev months: how many sales and how spread out their prices are.
    past = listings[listings["date_sold"] < "2025-06-01"]
    log_ppsf = np.log(past["sold_price"] / features.square_feet(past["house_area"]))
    recent = past["date_sold"] >= "2024-06-01"
    stats = pd.DataFrame({"community": past["community"], "recent": recent, "log_ppsf": log_ppsf})
    community = stats.groupby("community").agg(sales_12m=("recent", "sum"), ppsf_spread=("log_ppsf", "std"))
    return out.merge(community, left_on="community", right_index=True, how="left")


def by(frame, column):
    return frame.groupby(column, observed=True).apply(summary, include_groups=False).round(2)


def error_analysis():
    results = describe_homes(load_dev())
    sections = []
    for track in TRACKS:
        homes = results[results["track"] == track].copy()
        homes["sold price decile"] = pd.qcut(homes["sold_price"], 10, labels=range(1, 11))
        homes["predicted price decile"] = pd.qcut(homes["predicted_price"], 10, labels=range(1, 11))
        sections += [(f"{track}: overall", summary(homes).round(2).to_frame("value")),
                     (f"{track}: by property type", by(homes, "house_type_name")),
                     (f"{track}: by sold-price decile (1 = cheapest)", by(homes, "sold price decile")),
                     (f"{track}: by predicted-price decile", by(homes, "predicted price decile")),
                     (f"{track}: by horizon h", by(homes, "horizon")),
                     (f"{track}: by month", by(homes, "month")),
                     (f"{track}: by remarks condition score (-2 = needs major work)", by(homes, "condition")),
                     (f"{track}: house_area given as a range?", by(homes, "area_is_range"))]
        communities = homes.groupby("community").apply(summary, include_groups=False)
        communities = communities.join(homes.groupby("community")[["sales_12m", "ppsf_spread"]].first())
        communities = communities[communities["predictions"] >= 100].sort_values("MdAPE")
        sections += [(f"{track}: 10 hardest communities (>= 100 dev predictions)", communities.tail(10).round(3)),
                     (f"{track}: 10 easiest communities", communities.head(10).round(3)),
                     (f"{track}: community MdAPE vs history (Spearman)", pd.DataFrame({
                         "spearman with MdAPE": [communities["MdAPE"].corr(communities["sales_12m"], method="spearman"),
                                                 communities["MdAPE"].corr(communities["ppsf_spread"], method="spearman")]},
                         index=["sales in the 12 months before the dev period", "spread of log price/sqft"]).round(2))]
        if track == "list_aware":
            homes["list vs sold"] = pd.qcut(homes["list_vs_sold"], 5)
            sections.append(("list_aware: by how far the list price was from the sale (quintiles of log(list/sold))",
                             by(homes, "list vs sold")))
    write_report("error_analysis.md", sections)
    figures(results)


def figures(results):
    import matplotlib.pyplot as plt

    (ROOT / "figures").mkdir(exist_ok=True)
    style = dict(color="#e5e5e5")

    # 1. Error by property type, both tracks side by side.
    fig, ax = plt.subplots(figsize=(8, 4))
    kinds = (results[results["track"] == "off_market"].groupby("house_type_name")["ape"].median()
             .sort_values().index.tolist())
    x = np.arange(len(kinds))
    for i, track in enumerate(TRACKS):
        mdape = results[results["track"] == track].groupby("house_type_name")["ape"].median().reindex(kinds)
        bars = ax.bar(x + (i - 0.5) * 0.38, mdape, width=0.36, color=COLOURS[track], label=track)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, kinds, fontsize=9)
    ax.set_ylabel("Median absolute % error")
    ax.set_title("Dev error by property type (2025-06 to 2025-11, horizons 0-3)")
    ax.grid(axis="y", **style); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "error_by_type.png", dpi=150); plt.close(fig)

    # 2. Signed error by remarks condition score: does the text explain the misses?
    fig, ax = plt.subplots(figsize=(8, 4))
    for track in TRACKS:
        signed = results[results["track"] == track].groupby("condition")["signed"].median()
        ax.plot(signed.index, signed.values, marker="o", lw=2, color=COLOURS[track], label=track)
    ax.axhline(0, color="#888780", lw=1)
    ax.set_xlabel("Remarks condition score (-2 = needs major work, +3 = fully renovated)")
    ax.set_ylabel("Median signed % error\n(above 0: predicted too high)")
    ax.set_title("Dev signed error by condition described in the remarks")
    ax.grid(axis="y", **style); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "error_by_condition.png", dpi=150); plt.close(fig)

    # 3. Communities: how mixed their prices were vs error (off-market).
    homes = results[results["track"] == "off_market"]
    communities = homes.groupby("community").agg(mdape=("ape", "median"), n=("ape", "size"),
                                                 spread=("ppsf_spread", "first"))
    communities = communities[communities["n"] >= 100]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(communities["spread"], communities["mdape"], s=18, color=COLOURS["off_market"], alpha=0.8)
    ax.set_xlabel("Spread of log price per sq ft in the community before the dev period (std)")
    ax.set_ylabel("Off-market MdAPE (%)")
    ax.set_title("Off-market dev error by community (communities with >= 100 dev predictions)")
    ax.grid(**style); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "error_by_community.png", dpi=150); plt.close(fig)
    print("Wrote figures/error_by_type.png, figures/error_by_condition.png, figures/error_by_community.png")


# ---------------------------------------------------------------- test intervals

def test_intervals():
    """Calibrate on all dev predictions and put intervals around the submitted test predictions."""
    results = load_dev()
    for track in TRACKS:
        model = ConformalIntervals().fit(results[results["track"] == track])
        homes = data.load(track)
        submitted = pd.read_csv(ROOT / "submissions" / f"submission_{track}.csv")
        homes = homes[["ml_num", "house_type_name"]].merge(submitted, on="ml_num", validate="one_to_one")
        bounds = model.predict(homes["house_type_name"], homes["predicted_price"])
        out = pd.concat([homes.reset_index(drop=True), bounds.round(-2)], axis=1)
        path = ROOT / "outputs" / f"test_intervals_{track}.csv"
        out.to_csv(path, index=False)
        width = np.median(100 * (out["upper_80"] - out["lower_80"]) / out["predicted_price"])
        print(f"Wrote {path.relative_to(ROOT)} ({len(out):,} homes, median 80% width {width:.1f}% of the prediction)")


COMMANDS = {"intervals": interval_coverage, "errors": error_analysis, "test": test_intervals}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]]()
