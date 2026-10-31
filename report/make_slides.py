"""16:9 images for the video slides. Run from the repository folder after dev_validate.py final:

    python report/make_slides.py

Writes report/slides/slide{NN}{a,b}_*.png (1920 x 1080); NN = slide number in slides_outline.md,
which follows the report's sections. Step scores are copied from experiments.md.
If outputs/final_check_final.csv exists (the once-only final check), slide 6 includes it.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "slides"
BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#888780", "#0b0b0b", "#52514e", "#ecebe6"
LIGHT = {"history": "#d3d1c7", "dev": "#afa9ec", "final": "#f0997b", "test": "#5dcaa5"}
plt.rcParams.update({"font.size": 18, "axes.edgecolor": "#b4b2a9", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 22,
                     "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlepad": 14})


def slide(title):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
    fig.text(0.05, 0.92, title, fontsize=30, weight="bold", color=INK, va="center")
    return fig


def save(fig, name):
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def tidy(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load(name):
    results = pd.read_csv(ROOT / "outputs" / name)
    results["ape"] = 100 * (results["predicted_price"] - results["sold_price"]).abs() / results["sold_price"]
    results["signed"] = 100 * (results["predicted_price"] / results["sold_price"] - 1)
    return results


def summary(final):
    fig = slide("Summary: half the error of the baselines")
    ax = fig.add_axes([0.05, 0.5, 0.9, 0.28])
    # Months from 2021-01: data 0..65 (to May 2026), test 65..69 (Jun-Sep 2026).
    ax.barh(0, 65, color=LIGHT["history"], height=0.5)
    ax.barh(0, 4, left=65, color=LIGHT["test"], height=0.5)
    ax.text(32.5, 0, "150,000 sales we can learn from\nJan 2021 – May 2026", ha="center", va="center",
            fontsize=20, color=INK)
    ax.text(70, 0, "homes to predict\nJun–Sep 2026", ha="left", va="center", fontsize=16, color=INK)
    ax.annotate("", xy=(69, 0.4), xytext=(65, 0.4), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2.5))
    ax.text(64, 0.62, "0–4 months with no new data", ha="right", va="center", fontsize=17, color=ORANGE,
            weight="bold")
    ax.set_xlim(0, 85); ax.set_ylim(-0.5, 0.85); ax.axis("off")
    for x, label in [(0, "2021"), (12, "2022"), (24, "2023"), (36, "2024"), (48, "2025"), (60, "2026")]:
        ax.text(x, -0.42, label, ha="left", fontsize=15, color=MUTED)
    baselines = load("dev_predictions_passline_starter_feature.csv")
    md = lambda f, t, m=None: f[(f["track"] == t) & ((f["model"] == m) if m else True)]["ape"].median()
    tiles = [(0.05, "Off-market (no list price)", BLUE, md(final, "off_market"),
              md(baselines, "off_market", "passline"), "pass line"),
             (0.52, "List-aware (list price known)", ORANGE, md(final, "list_aware"),
              md(baselines, "list_aware", "starter"), "list price alone")]
    for x, head, colour, ours, base, base_label in tiles:
        fig.patches.append(FancyBboxPatch((x, 0.07), 0.43, 0.33, boxstyle="round,pad=0.005,rounding_size=0.015",
                                          transform=fig.transFigure, facecolor="#f7f6f2", edgecolor=colour, lw=2.5))
        fig.text(x + 0.02, 0.34, head, fontsize=20, weight="bold", color=colour)
        fig.text(x + 0.02, 0.17, f"{ours:.2f}%", fontsize=48, weight="bold", color=INK)
        fig.text(x + 0.22, 0.2, f"vs {base:.2f}%\n{base_label}", fontsize=18, color=MUTED)
        fig.text(x + 0.02, 0.1, "median error, dev months, 0–3 months ahead", fontsize=14, color=MUTED)
    save(fig, "slide03_summary.png")


def models_slide():
    fig = slide("Two models, two targets")
    rows = [(0.66, BLUE, "Off-market", "home + remarks + market", "1,500 trees", "log(price)", "price"),
            (0.36, ORANGE, "List-aware", "same + list price, past\nsale-to-list, list vs market", "100 trees",
             "log(sold / list)", "× list price")]
    for y, colour, name, inputs, trees, target, out in rows:
        fig.text(0.05, y + 0.06, name, fontsize=24, weight="bold", color=colour)
        for x, w, text in [(0.05, 0.27, inputs), (0.38, 0.16, trees), (0.6, 0.17, target), (0.83, 0.12, out)]:
            fig.patches.append(FancyBboxPatch((x, y - 0.1), w, 0.13, boxstyle="round,pad=0.005,rounding_size=0.012",
                                              transform=fig.transFigure, facecolor="#f7f6f2", edgecolor=colour, lw=2))
            fig.text(x + w / 2, y - 0.035, text, ha="center", va="center", fontsize=17, color=INK)
        for x in [0.325, 0.545, 0.775]:
            fig.text(x + 0.012, y - 0.035, "→", ha="center", va="center", fontsize=26, color=MUTED)
    fig.text(0.05, 0.06, "Both: gradient-boosted trees (scikit-learn). A sale counts half as much for every year it is older.\n"
             "Market windows end at the listing date, or at the last day of data for homes listed later.", fontsize=15, color=MUTED)
    save(fig, "slide05_models.png")


def limitations(final):
    fig = slide("Both models still price a little high")
    ax = fig.add_axes([0.1, 0.18, 0.5, 0.6])
    for track, colour, label in [("off_market", BLUE, "off-market"), ("list_aware", ORANGE, "list-aware")]:
        signed = final[final["track"] == track].groupby("horizon")["signed"].median()
        ax.plot(signed.index, signed.values, color=colour, lw=3, marker="o", ms=8)
        ax.text(3.12, signed.values[-1], label, va="center", fontsize=16, color=colour)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(4), ["0", "1", "2", "3"]); ax.set_xlim(-0.2, 4.2); ax.set_ylim(-1, 6)
    ax.set_xlabel("months between data end and sale"); ax.set_ylabel("median (predicted − sold) / sold (%)")
    tidy(ax)
    notes = ["Prices kept falling in 2025:", "the further ahead,", "the higher we price.", "",
             "A bias correction helped", "on dev, but at our May 2026", "data end it would raise", "every prediction by 3.4%,",
             "a direction never validated", "→ left out."]
    for i, line in enumerate(notes):
        fig.text(0.68, 0.74 - i * 0.055, line, fontsize=17, color=INK)
    save(fig, "slide09_limitations.png")


def validation(final):
    fig = slide("We tested the way the real test works")
    left = fig.add_axes([0.05, 0.14, 0.5, 0.66])
    spans = [(0, 53, LIGHT["history"], "training history"), (53, 59, LIGHT["dev"], "dev: every decision"),
             (59, 65, LIGHT["final"], "final check: run once"), (65, 69, LIGHT["test"], "hidden test")]
    for a, b, colour, label in spans:
        left.barh(0, b - a, left=a, color=colour, height=0.45, edgecolor="white", lw=2)
    from matplotlib.patches import Patch
    left.legend([Patch(color=c) for *_, c, _ in spans], [label for *_, label in spans], ncol=2, fontsize=14,
                frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    for h in range(4):
        y = 0.9 + 0.36 * h
        left.barh(y, 56 - h, left=0, color="#e3e1d9", height=0.24)
        left.barh(y, 1, left=56, color="#7f77dd", height=0.24)
        left.text(57.6, y, f"data ends {h} month{'s' if h != 1 else ''} earlier" if h else "data ends just before",
                  va="center", fontsize=13, color=MUTED)
    left.text(0, 2.5, "Each dev month is predicted 4 times (here: Sep 2025)", fontsize=15, color=INK)
    left.set_xlim(0, 92); left.set_ylim(-0.4, 2.75); left.set_yticks([])
    left.set_xticks([0, 12, 24, 36, 48, 60], ["2021", "2022", "2023", "2024", "2025", "2026"], fontsize=14)
    left.spines[["top", "right", "left"]].set_visible(False)
    right = fig.add_axes([0.68, 0.2, 0.28, 0.58])
    baselines = load("dev_predictions_passline_starter_feature.csv")
    for label, rows, colour, ls in [("pass line", baselines[baselines["model"] == "passline"], GRAY, "--"),
                                    ("starter trees", baselines[baselines["model"] == "feature"], ORANGE, "-"),
                                    ("ours", final, BLUE, "-")]:
        mdape = rows[rows["track"] == "off_market"].groupby("horizon")["ape"].median()
        right.plot(mdape.index, mdape.values, color=colour, ls=ls, lw=3, marker="o", ms=7)
        right.text(3.15, mdape.values[-1], label, va="center", fontsize=15, color=INK)
    right.set_xticks(range(4), ["0", "1", "2", "3"]); right.set_xlim(-0.2, 4.6); right.set_ylim(0, 22)
    right.set_xlabel("months between data end and sale"); right.set_ylabel("off-market MdAPE (%)")
    right.set_title("Error vs months ahead", fontsize=17)
    tidy(right)
    save(fig, "slide06_validation.png")


def information():
    fig = slide("Three kinds of information")
    columns = [("The home", BLUE, ["floor area", "bedrooms, bathrooms", "parking, garage",
                                   "lot size (houses)", "type, style, location"]),
               ("The remarks", "#7f77dd", ["\"fully renovated\"", "\"new roof / furnace\"", "\"finished basement\"",
                                           "\"dated\", \"needs work\"", "→ 0/1 flags + score"]),
               ("The market", ORANGE, ["price per sq ft nearby", "same type & community", "last 1–12 months",
                                       "only sales before listing", "recent sales count more"])]
    for i, (head, colour, items) in enumerate(columns):
        x = 0.05 + i * 0.31
        fig.patches.append(FancyBboxPatch((x, 0.1), 0.28, 0.7, boxstyle="round,pad=0.005,rounding_size=0.015",
                                          transform=fig.transFigure, facecolor="#f7f6f2", edgecolor=colour, lw=2.5))
        fig.text(x + 0.02, 0.73, head, fontsize=24, weight="bold", color=colour)
        for j, item in enumerate(items):
            fig.text(x + 0.02, 0.62 - j * 0.1, item, fontsize=17, color=INK)
    save(fig, "slide04a_information.png")


def condition():
    """Remarks condition score vs price, on sales before the dev months (experiments.md #9)."""
    fig = slide("What the remarks say is worth money")
    ax = fig.add_axes([0.12, 0.17, 0.82, 0.62])
    scores, effect = [-2, -1, 0, 1, 2, 3], [-7.2, -4.4, -0.5, 3.4, 8.4, 9.0]
    ax.bar(scores, effect, color=[ORANGE if e < 0 else BLUE for e in effect], width=0.6)
    for s, e in zip(scores, effect):
        ax.text(s, e + (0.5 if e >= 0 else -0.5), f"{e:+.1f}%", ha="center", va="bottom" if e >= 0 else "top",
                fontsize=17, color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(scores, ["−2\ne.g. \"needs work\"", "−1", "0", "+1", "+2", "+3\ne.g. \"fully redone\""])
    ax.set_ylabel("price per sq ft vs\nneighbourhood (%)", fontsize=16); ax.set_ylim(-10, 12)
    ax.set_xlabel("net condition score from the remarks (sales before Jun 2025)")
    tidy(ax)
    save(fig, "slide04b_condition.png")


def ablation():
    fig = slide("Each idea, added one at a time")
    off = [("starter trees", 15.78), ("+ property features", 14.63), ("+ listing month", 14.55),
           ("+ past market prices", 11.56), ("+ remarks phrases", 11.04), ("+ windows end at data end", 10.77),
           ("+ recent sales weigh more", 10.58), ("+ larger trees", 9.19)]
    lst = [("trees + list price", 4.71), ("+ predict sold / list", 4.68), ("+ past sale-to-list", 4.46),
           ("+ list vs market", 4.32), ("+ recent sales weigh more", 4.24)]
    for (x, w), steps, colour, ref, ref_label, title, hi in [
            ((0.25, 0.3), off, BLUE, 18.70, "pass line", "Off-market", 3),
            ((0.73, 0.23), lst, ORANGE, 7.09, "list price", "List-aware", 2)]:
        ax = fig.add_axes([x, 0.2, w, 0.58])
        labels, values = zip(*steps)
        ys = np.arange(len(steps))[::-1]
        bars = ax.barh(ys, values, color=colour, height=0.62, alpha=0.45)
        bars[hi].set_alpha(1.0)
        for y, v in zip(ys, values):
            ax.text(v + ref * 0.015, y, f"{v:.2f}", va="center", fontsize=14, color=INK)
        ax.axvline(ref, color=GRAY, ls="--", lw=1.5)
        ax.text(ref, len(steps) - 0.3, f"{ref_label} {ref:.2f}", ha="right", fontsize=13, color=MUTED)
        ax.set_ylim(-0.6, len(steps) + 0.1); ax.set_xlim(0, ref * 1.15)
        ax.set_yticks(ys, labels, fontsize=14); ax.set_xlabel("dev MdAPE (%)", fontsize=15)
        ax.set_title(title, fontsize=19); tidy(ax, "x")
    fig.text(0.05, 0.04, "Tried and dropped: simulated stale data in training (no gain; worse with large trees),\n"
             "trimming outliers, bias correction, larger list-aware trees.", fontsize=14, color=MUTED)
    save(fig, "slide07b_ablation.png")


def results(final):
    fig = slide("About half the error of the baselines")
    baselines = load("dev_predictions_passline_starter_feature.csv")
    check_path = ROOT / "outputs" / "final_check_final.csv"
    check = load("final_check_final.csv") if check_path.exists() else None

    def md(frame, track, model=None):
        rows = frame[(frame["track"] == track) & ((frame["model"] == model) if model else True)]
        return f"{rows['ape'].median():.2f}%"
    rows = [["", "Off-market", "List-aware"],
            ["Baseline", f"{md(baselines, 'off_market', 'passline')} (pass line)",
             f"{md(baselines, 'list_aware', 'starter')} (list price)"],
            ["Ours, dev months (Jun–Nov 2025)", md(final, "off_market"), md(final, "list_aware")],
            ["Ours, final check (Dec 2025–May 2026)",
             md(check, "off_market") if check is not None else "[pending]",
             md(check, "list_aware") if check is not None else "[pending]"]]
    ax = fig.add_axes([0.05, 0.3, 0.9, 0.46]); ax.axis("off")
    table = ax.table(cellText=rows, cellLoc="left", colWidths=[0.48, 0.26, 0.26], loc="center")
    table.auto_set_font_size(False); table.set_fontsize(18); table.scale(1, 3.2)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddcd6"); cell.set_linewidth(0.8)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK); cell.set_facecolor("#f1f0eb")
        if r >= 2 and c > 0:
            cell.set_text_props(weight="bold", color=BLUE if c == 1 else ORANGE)
    fig.text(0.05, 0.14, "MdAPE: median of |predicted − sold| / sold, every home predicted with data ending "
             "0–3 months earlier.", fontsize=15, color=MUTED)
    save(fig, "slide07a_results.png")


def where_it_fails(final):
    import data
    import features
    fig = slide("Where we are wrong, and why")
    listings = data.load("listings")
    past = listings[listings["date_sold"] < "2025-06-01"]
    spread = (np.log(past["sold_price"] / features.square_feet(past["house_area"]))
              .groupby(past["community"]).std().rename("spread"))
    homes = final.merge(listings[["ml_num", "community"]], on="ml_num")
    left = fig.add_axes([0.07, 0.2, 0.46, 0.58])
    off = homes[homes["track"] == "off_market"]
    kinds = off.groupby("house_type_name")["ape"].median().sort_values().index
    short = {"Condo Townhouse": "Condo TH", "Condo Apt": "Condo apt", "Detached": "Detached",
             "Semi-Detached": "Semi", "Co-Op Apt": "Co-op apt", "Freehold Townhouse": "Freehold TH"}
    x = np.arange(len(kinds))
    for i, (track, colour) in enumerate([("off_market", BLUE), ("list_aware", ORANGE)]):
        mdape = homes[homes["track"] == track].groupby("house_type_name")["ape"].median().reindex(kinds)
        bars = left.bar(x + (i - 0.5) * 0.4, mdape, width=0.38, color=colour, label=track.replace("_", "-"))
        left.bar_label(bars, fmt="%.1f", fontsize=12, padding=2)
    left.set_xticks(x, [short[k] for k in kinds], fontsize=14, rotation=20, ha="right")
    left.set_ylabel("dev MdAPE (%)"); left.legend(frameon=False, fontsize=14)
    left.set_title("Houses are harder than condos", fontsize=18); tidy(left)
    right = fig.add_axes([0.62, 0.2, 0.34, 0.58])
    by_community = off.groupby("community").agg(mdape=("ape", "median"), n=("ape", "size")).join(spread)
    by_community = by_community[by_community["n"] >= 100]
    right.scatter(by_community["spread"], by_community["mdape"], s=40, color=BLUE, alpha=0.8)
    rho = by_community["mdape"].corr(by_community["spread"], method="spearman")
    right.set_xlabel("how mixed past prices were (std)"); right.set_ylabel("off-market MdAPE (%)")
    right.set_title("Mixed neighbourhoods are harder", fontsize=18)
    right.text(0.03, 0.95, f"Spearman ρ = {rho:.2f}", transform=right.transAxes, fontsize=15, color=MUTED, va="top"); tidy(right, "both")
    save(fig, "slide08_where_it_fails.png")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    OUT.mkdir(exist_ok=True)
    information(); condition(); models_slide(); ablation()
    if "--static" not in sys.argv:
        final = load("dev_predictions_final.csv")
        summary(final); validation(final); results(final); where_it_fails(final); limitations(final)
    print(f"Wrote {OUT}")
