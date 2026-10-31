# SSU Data Dash: predict Toronto home prices

Start with the case handout, [`handout.md`](handout.md), and the judging rubric, [`rubric.md`](rubric.md).

Two tracks:
- **Off-market:** value a home with no list price.
- **List-aware:** predict a listed home's sale price, knowing its list price.

Train on 150,000 sales (January 2021 to May 2026). Predict homes that sold June to September 2026.

**Data:** [huggingface.co/datasets/Baliyang/SSU_data_dash](https://huggingface.co/datasets/Baliyang/SSU_data_dash).
All of it is synthetic.

## Our models (team)

Both tracks use gradient-boosted trees (`FeatureModel` in `models.py`) on features built only from
sales that closed before each home was listed (`features.py`). Off-market learns log(price);
list-aware learns log(sold / list). The experiment log, with every dev result and decision, is
[`experiments.md`](experiments.md).

Reproduce the two submission files from a fresh copy (Python 3.12, about 2 minutes):

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q           # includes leakage checks on the features
.venv/bin/python run.py predict         # writes submissions/ (deterministic: same bytes every run)
.venv/bin/python run.py check
```

Our extra commands:

```
python dev_validate.py final   # dev validation: Jun-Nov 2025 at 0-3 month gaps (~15 min)
python phase3.py intervals     # 80%/90% interval coverage on held-out dev months (needs the line above)
python phase3.py errors        # error tables (reports/) and figures (figures/)
python phase3.py test          # interval files for the test homes (after run.py predict)
```

Team rules: tune and select only with `dev_validate.py`; `run.py validate` (Dec 2025 - May 2026)
is our final check and is run once, after the models are frozen. The off-market model never sees
`list_price`. New aggregate features use only sales before the listing date and get a test.

## Setup

Python 3.10 or newer (3.11+ on macOS 27):

```
pip install -r requirements.txt
```

The data downloads itself on first use. No Python? Use the
[Colab notebook](https://colab.research.google.com/github/erwinzhang7/SSU-Data-Dash/blob/main/notebooks/quickstart.ipynb).
Prefer R? See `r/baseline.R`.

## Commands

```
python run.py validate   # test on Dec 2025 to May 2026
python run.py report     # results table and a figure
python run.py predict    # write both submission files
python run.py check      # check them
```

On Windows, use `py` if `python` is not found.

## What to edit

- **`features.py`**: the features.
- **`models.py`**: the models. Two simple baselines to beat: list price (list-aware) and the
  10 nearest earlier sales (off-market). `FeatureModel` (gradient-boosted trees) is ready to switch on.

## Validation

`validate` predicts each month from the sales before it, as in the real test. A random split lets
the model see the future and makes your score look better than it is. Judges compare your
validation error with your hidden-test error.

## Submission

One PDF report (template: `report/report.tex`, 4 pages max) that links to:
1. your video on YouTube, 5 minutes max (unlisted is fine, not private);
2. your GitHub repository (public, or shared with the organisers), holding your code and two files:

```
submissions/submission_off_market.csv    one row per home in test_off_market
submissions/submission_list_aware.csv    one row per home in test_list_aware

ml_num,predicted_price
SYN-0123456,815000
```

`predict` writes both; run `check` before you hand in. Deadline: see the case email.

## Rules

- Outside data only if it covers May 2026 or earlier. List every source with its URL.
- Neighbourhoods, coordinates and streets are synthetic, so map data will not line up.
- No real listing or sales records.
- Do not try to recover the hidden test prices.
