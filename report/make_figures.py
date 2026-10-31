"""Figures and the dev results table for report.tex. Run from the repository folder:

    python dev_validate.py final passline starter feature   (once, ~25 min; or reuse outputs/)
    python report/make_figures.py

Step-by-step dev scores are copied from experiments.md (each needed its own dev run).
Writes report/figures/*.png and report/dev_results.tex.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#888780", "#0b0b0b", "#52514e", "#ecebe6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b4b2a9", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.5, "axes.titlelocation": "left"})


def tidy(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def load(name):
    results = pd.read_csv(ROOT / "outputs" / f"dev_predictions_{name}.csv")
    results["ape"] = 100 * (results["predicted_price"] - results["sold_price"]).abs() / results["sold_price"]
    results["signed"] = 100 * (results["predicted_price"] / results["sold_price"] - 1)
    return results


def validation_design(baselines, final):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})
    # Left: which months do what. 2021-01 .. 2026-09 = 69 months.
    spans = [(0, 53, "#d3d1c7", "training history"), (53, 59, "#afa9ec", "dev (all choices)"),
             (59, 65, "#f0997b", "final check (once)"), (65, 69, "#5dcaa5", "hidden test")]
    from matplotlib.patches import Patch
    for a, b, colour, label in spans:
        left.barh(0, b - a, left=a, color=colour, height=0.5, edgecolor="white", lw=1.5)
    left.legend([Patch(color=c) for _, _, c, _ in spans], [label for *_, label in spans], ncol=2, fontsize=7,
                frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.18), handlelength=1.2)
    for i, h in enumerate(range(4)):
        y = 1.0 + 0.32 * i
        left.barh(y, 56 - h, left=0, color="#e3e1d9", height=0.2)
        left.barh(y, 1, left=56, color="#7f77dd", height=0.2)
        left.text(57.5, y, f"h={h}", va="center", fontsize=7, color=MUTED)
    left.text(0, 2.45, "Dev: Sep 2025 predicted 4 times, data cut 0-3 months earlier",
              fontsize=7.5, color=INK)
    left.set_xlim(0, 75); left.set_ylim(-0.4, 2.7); left.set_yticks([])
    left.set_xticks([0, 12, 24, 36, 48, 60], ["2021", "2022", "2023", "2024", "2025", "2026"])
    left.spines[["top", "right", "left"]].set_visible(False)
    left.set_title("(a) Months and their roles")
    # Right: off-market dev MdAPE by horizon.
    series = [("pass line", baselines[baselines["model"] == "passline"], GRAY, "--"),
              ("starter trees", baselines[baselines["model"] == "feature"], ORANGE, "-"),
              ("ours", final, BLUE, "-")]
    for label, rows, colour, ls in series:
        mdape = rows[rows["track"] == "off_market"].groupby("horizon")["ape"].median()
        right.plot(mdape.index, mdape.values, color=colour, ls=ls, lw=1.8, marker="o", ms=3.5)
        right.text(3.12, mdape.values[-1], label, va="center", fontsize=7.5, color=INK)
    right.set_xticks(range(4), ["h=0", "1", "2", "3"]); right.set_xlim(-0.2, 4.4); right.set_ylim(0, 21)
    right.set_ylabel("Off-market MdAPE (%)"); tidy(right)
    right.set_title("(b) Error vs months since data end")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "validation_design.png", dpi=220); plt.close(fig)


def ablation():
    off = [("starter trees", 15.78), ("+ property features", 14.63), ("+ listing month", 14.55),
           ("+ past market levels", 11.56), ("+ remarks phrases", 11.04), ("+ windows end at data end", 10.77),
           ("+ recency weights", 10.58), ("+ larger trees", 9.19)]
    lst = [("trees + list price", 4.71), ("+ target log(sold/list)", 4.68), ("+ past sale-to-list", 4.46),
           ("+ list vs market", 4.32), ("+ recency weights", 4.24)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, steps, colour, ref, ref_label, title in [
            (axes[0], off, BLUE, 18.70, "pass line", "(a) Off-market"),
            (axes[1], lst, ORANGE, 7.09, "list price", "(b) List-aware")]:
        labels, values = zip(*steps)
        ys = np.arange(len(steps))[::-1]
        ax.barh(ys, values, color=colour, height=0.6)
        for y, v in zip(ys, values):
            ax.text(v + ref * 0.015, y, f"{v:.2f}", va="center", fontsize=7, color=INK)
        ax.axvline(ref, color=GRAY, ls="--", lw=1)
        ax.text(ref, len(steps) - 0.25, f"{ref_label} {ref:.2f}", ha="right", fontsize=7, color=MUTED)
        ax.set_ylim(-0.6, len(steps) + 0.1)
        ax.set_yticks(ys, labels, fontsize=7.5); ax.set_xlim(0, ref * 1.12)
        ax.set_xlabel("Dev MdAPE (%)"); tidy(ax, "x"); ax.set_title(title)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "ablation.png", dpi=220); plt.close(fig)


def where_it_fails(final):
    import data
    import features
    listings = data.load("listings")
    past = listings[listings["date_sold"] < "2025-06-01"]
    spread = (np.log(past["sold_price"] / features.square_feet(past["house_area"]))
              .groupby(past["community"]).std().rename("spread"))
    homes = final.merge(listings[["ml_num", "community"]], on="ml_num")
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.25, 1]})
    off = homes[homes["track"] == "off_market"]
    kinds = off.groupby("house_type_name")["ape"].median().sort_values().index
    x = np.arange(len(kinds))
    short = {"Condo Townhouse": "Condo TH", "Condo Apt": "Condo apt", "Detached": "Detached",
             "Semi-Detached": "Semi", "Co-Op Apt": "Co-op apt", "Freehold Townhouse": "Freehold TH"}
    for i, (track, colour) in enumerate([("off_market", BLUE), ("list_aware", ORANGE)]):
        mdape = homes[homes["track"] == track].groupby("house_type_name")["ape"].median().reindex(kinds)
        bars = left.bar(x + (i - 0.5) * 0.4, mdape, width=0.38, color=colour, label=track.replace("_", "-"))
        left.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=1)
    left.set_xticks(x, [short[k] for k in kinds], fontsize=7, rotation=25, ha="right")
    left.set_ylabel("Dev MdAPE (%)")
    left.legend(frameon=False, fontsize=7.5); tidy(left); left.set_title("(a) By property type")
    by_community = off.groupby("community").agg(mdape=("ape", "median"), n=("ape", "size")).join(spread)
    by_community = by_community[by_community["n"] >= 100]
    right.scatter(by_community["spread"], by_community["mdape"], s=10, color=BLUE, alpha=0.8)
    rho = by_community["mdape"].corr(by_community["spread"], method="spearman")
    right.set_xlabel("Past spread of log price/sq ft (std)")
    right.set_ylabel("Off-market MdAPE (%)"); tidy(right, "both")
    right.set_title(f"(b) By community (Spearman {rho:.2f})")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "where_it_fails.png", dpi=220); plt.close(fig)


def dev_table(baselines, final):
    rows = []
    names = {("off_market", "passline"): "Pass line (community x type, 12 months)",
             ("off_market", "starter"): "Starter: 10 nearest earlier sales",
             ("off_market", "final"): "\\textbf{Ours}",
             ("list_aware", "starter"): "Starter: list price",
             ("list_aware", "final"): "\\textbf{Ours}"}
    frames = pd.concat([baselines, final])
    for (track, model), label in names.items():
        r = frames[(frames["track"] == track) & (frames["model"] == model)]
        by_h = r.groupby("horizon")["ape"].median()
        rows.append(f"{track.replace('_', '-')} & {label} & {r['ape'].median():.2f} & {r['ape'].mean():.2f} & "
                    f"{r['signed'].median():+.2f} & {by_h[0]:.2f} & {by_h[3]:.2f} \\\\")
    table = "\n".join([
        "% Written by report/make_figures.py from outputs/dev_predictions_*.csv. Rerun instead of editing.",
        "\\begin{tabular}{llrrrrr}", "\\toprule",
        "Track & Model & MdAPE & MAPE & Signed & MdAPE $h{=}0$ & MdAPE $h{=}3$ \\\\", "\\midrule",
        rows[0], rows[1], rows[2], "\\midrule", rows[3], rows[4], "\\bottomrule", "\\end{tabular}"])
    (ROOT / "report" / "dev_results.tex").write_text(table + "\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    OUT.mkdir(exist_ok=True)
    baselines = load("passline_starter_feature")
    final = load("final")
    validation_design(baselines, final)
    ablation()
    where_it_fails(final)
    dev_table(baselines, final)
    print("Wrote report/figures/*.png and report/dev_results.tex")
