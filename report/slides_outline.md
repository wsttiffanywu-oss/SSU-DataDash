# Slide outline (11 slides, follows the report)

What goes ON each slide: the title, the few words on screen, and the image. Keep on-screen text
short — the narration (video_script.md) carries the detail. Images are in report/slides/
(1920 × 1080, made by `python report/make_slides.py`). Numbers in 【】 are filled in after the
final check.

---

## Slide 1 — Title (0:00–0:10)

**Title:** Valuing Toronto homes 0–4 months ahead

**On screen:** SSU Data Dash 2026 · 【team name】 · 【full names】

**Image:** none.

## Slide 2 — Table of contents (0:10–0:20)

**Title:** Table of Contents

**On screen:** Summary · Data and Features · Models · Validation · Results · Where the model fails ·
Limitations · Outside Data · Reproducing our results

**Image:** none (your template).

## Slide 3 — Summary (0:20–0:50)

**Title:** Summary

**On screen:** Data ends May 2026; homes sell Jun–Sep 2026 → predict 0–4 months ahead.
Off-market 9.19% vs 18.70% pass line. List-aware 4.24% vs 7.09% list price.

**Image:** slide03_summary.png (timeline + the two headline numbers).

## Slide 4 — Data and Features (0:50–1:35)

**Title:** Data and Features

**On screen:** Competition data only. The home · the remarks · the market. Every market number uses
only sales before the home was listed (unit-tested).

**Image:** slide04a_information.png (three columns). Optional second image or build:
slide04b_condition.png (remarks condition score vs price).

## Slide 5 — Models (1:35–2:05)

**Title:** Models

**On screen:** Gradient-boosted trees. Off-market learns log(price). List-aware learns log(sold /
list), × list price. Recent sales weigh more.

**Image:** slide05_models.png.

## Slide 6 — Validation (2:05–2:50)

**Title:** Validation

**On screen:** Walk-forward, never random. Each dev month predicted 4 times (data ending 0–3 months
earlier). Dev Jun–Nov 2025: every decision, 35 logged experiments. Final check Dec 2025–May
2026: run once, after freezing.

**Image:** slide06_validation.png (months and their roles + error vs months ahead).

## Slide 7 — Results (2:50–3:35)

**Title:** Results

**On screen:** Results table (dev + final check). Biggest gain: past market prices (−3.0 points).
80% intervals covered 【XX】% of held-out sales.

**Image:** slide07a_results.png (table). Second image or build: slide07b_ablation.png (each idea
added one at a time).

## Slide 8 — Where the model fails (3:35–4:10)

**Title:** Where the model fails

**On screen:** Condos ~8%, houses 11–12%. Mixed neighbourhoods, not small ones. List-aware: bidding
wars and deep discounts.

**Image:** slide08_where_it_fails.png.

## Slide 9 — Limitations (4:10–4:30)

**Title:** Limitations

**On screen:** Still ~3% high (falling 2025 market). Bias correction tested, left out. Hand-picked
remark phrases. Only gradient-boosted trees.

**Image:** slide09_limitations.png (overestimate by months ahead).

## Slide 10 — Outside Data (4:30–4:35)

**Title:** Outside Data

**On screen:** None. Competition files only.

**Image:** none.

## Slide 11 — Reproducing our results (4:35–4:50)

**Title:** Reproducing our results

**On screen:**
`pip install -r requirements.txt` · `pytest` · `python run.py predict` · `python run.py check`.
Fixed seeds, pinned versions → byte-identical files. 【GitHub link】

**Image:** none (commands on screen), or a QR code to the repository.
