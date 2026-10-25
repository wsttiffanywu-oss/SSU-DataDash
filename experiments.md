# Experiment log

Dev validation = `python dev_validate.py`: months 2025-06 to 2025-11, each scored at horizons
0-3 (train only on sales closed before the start of month − h), MdAPE/MAPE pooled over all horizons.
Final validation = `python run.py validate` (2025-12 to 2026-05, horizon 0). Touch it rarely.

| # | Date | Change | Track | Dev MdAPE | Dev MAPE | Final val touched? | Notes |
|---|---|---|---|---|---|---|---|
| 0 | 2026-10-23 | Starter baselines, reproduce official numbers | off / list | — | — | Yes (reproduction only) | 20.84 / 27.43 off; 7.26 / 8.09 list. Matches handout. |
| 1 | 2026-10-23 | Pass line (community×type 12-mo median) | both | 18.70 | 23.40 | No | Median signed +6.7%. h0→h3: 18.50→18.91 |
| 2 | 2026-10-23 | Starter baselines on dev | off | 21.00 | 28.10 | No | signed +9.9% |
|   |            |                          | list | 7.09 | 7.92 | No | signed +5.8% (homes closing below list) |
| 3 | 2026-10-23 | Starter `FeatureModel` (HGB, log price, starter features) | off | 15.78 | 20.38 | No | signed +5.6%; h0→h3: 15.27→16.34 |
|   |            |                                                           | list | 4.82 | 6.02 | No | signed +2.1%; h0→h3: 4.66→5.01 |
| 4 | 2026-10-23 | Phase 1 step 1: + area_is_range, lot_front/depth/area, garage, house_style, municipality (categoricals via train vocabulary) | off | 14.63 | 18.93 | No | −1.15 vs #3. signed +5.6% unchanged; h0→h3: 14.15→14.92 |
|   |            |                                                           | list | 4.78 | 5.99 | No | −0.04 vs #3 |
| 5 | 2026-10-23 | #4 + drop training sales outside type's 0.1–99.9 pct price | off | 14.61 | 18.92 | No | No real change vs #4 → **not adopted** |
|   |            |                                                           | list | 4.81 | 6.02 | No | slightly worse |
| 6 | 2026-10-23 | Phase 1 step 2: #4 + list_month (calendar month of date_listed) | off | 14.55 | 18.87 | No | −0.08 vs #4, same direction at every horizon (h0→h3: 14.13→14.87). Signed bias not fixed: overall +5.7%; Jun 4.4→5.9, Jul 7.2→7.1, Aug 7.8→7.3. Kept for now (tentative) |
|   |            |                                                           | list | 4.73 | 5.92 | No | −0.05 vs #4 |

**Diagnosis after #6.** Off-market signed error by horizon h=0/1/2/3: +4.2 / +5.2 / +6.3 / +7.3%. The bias grows ~1 point per month of gap (the market kept falling after the training cutoff = trend), and there is already +4% at h=0 (the model's price level lags the recent market). Condo Apt, with little seasonality, is also biased +3–5%. So the bias is mostly level + trend, not seasonality → steps 3 and 5.

| # | Date | Change | Track | Dev MdAPE | Dev MAPE | Final val touched? | Notes |
|---|---|---|---|---|---|---|---|
| 7 | 2026-10-23 | Phase 1 step 3: #6 + market_features(): mean log(price/sqft) of sales closed strictly before date_listed, by community×type (3/6/12 mo), community (3/12), type city-wide (1/3/12), all city (1/3/12); counts; momentum (3 vs 12 mo); location premium; naive log price (ct 6-mo ppsf × sqft). Leakage tests in tests/test_features.py | off | 11.56 | 14.91 | No | −2.99 vs #6. h0→h3: 10.98→12.15. Signed +4.3% (h0 +2.4 → h3 +6.0) |
|   |            |                                                           | list | 4.95 | 6.09 | No | **+0.22 worse** vs #6: h0 better (4.54) but h2/h3 worse (5.17/5.26). Stale market features compete with the fresh list price |

Leakage review for #7: unit tests show a home's features ignore its own sale and every sale closing on/after its listing date, and match a direct calculation. In dev_validate the history is only the training slice (before cutoff). Error still grows with horizon (10.98→12.15) and bias still +6% at h=3, which a leak would hide.

| # | Date | Change | Track | Dev MdAPE | Dev MAPE | Final val touched? | Notes |
|---|---|---|---|---|---|---|---|
| 8 | 2026-10-23 | List-aware: drop market features again (property features only + list_price) | list | 4.73 | 5.92 | No | Same as #6, as expected. Market features stay off-market only until Phase 2 |
| 9 | 2026-10-23 | Phase 1 step 4: + remark_features() — 8 phrase groups (renovated, luxury finish, new systems, finished basement, landscaped, dated, major repairs, motivated seller) as 0/1, net condition score, text length. Phrases chosen from sales before 2025-06 only | off | 11.04 | 14.06 | No | −0.52 vs #7. h0→h3: 10.40→11.75; signed h0→h3 +2.5→+6.1 |
|   |            |                                                           | list | 4.71 | 5.88 | No | −0.02 vs #8 |

Remarks leakage check (#9): no "$", no price figures ("899,000", "1.2m"), no "asking"/"reduced"/"priced" in any remark (the only "pric…" hits are 6 × "priceless" view), so the list price does not leak into the off-market model through text. "Seller is motivated / bring your offer" (5.8% of homes) goes with a lower sold/list ratio (0.975 vs 0.988); it is in the test files, so it is a legitimate feature.
Condition score on pre-dev sales vs. neighbourhood-quarter mean log price/sqft: −2: −7.2%, −1: −4.4%, 0: −0.5%, +1: +3.4%, +2: +8.4%, +3: +9.0% — monotonic.

| 10 | 2026-10-23 | Step 5a `anchor_only`: market windows end at min(date_listed, last known sale + 1 day) instead of date_listed, so homes listed after the data ends get the latest windows instead of empty ones | off | 10.77 | 13.68 | No | −0.27 vs #9. signed h0→h3 +2.5→+5.1 |
| 11 | 2026-10-23 | Step 5b `stale120`: #10 + training rows get a random as_of 0–120 days before listing + `staleness_days` feature | off | 10.74 | 13.63 | No | −0.03 vs #10 (small); signed h0→h3 +2.1→+5.0 |
| 12 | 2026-10-23 | Step 5c `stale120_hl365`: #11 + sample weight halves every 365 days of age | off | **10.53** | 13.47 | No | −0.21 vs #11; signed h0→h3 +2.0→+4.4 |
| 13 | 2026-10-23 | `stale120_hl730`: half-life 730 days | off | 10.62 | 13.54 | No | between #11 and #12 |
| 14 | 2026-10-23 | `stale120_nomonths`: #11 without `months` | off | 11.66 | 14.72 | No | +0.92 worse, signed +6.5% → keep `months` |
| 15 | 2026-10-23 | `stale120_hl180`: half-life 180 days | off | 10.74 | 13.71 | No | worse than 365 → 365 kept |
| 16 | 2026-10-23 | `stale120_hl365_bc`: #12 + bias correction (FeatureModel.fit_bias: train a copy on all but the last 120 days of training, predict those days, remove the median log error) | off | 10.38 | 13.08 | No | −0.15 vs #12; signed overall +0.9% (h0→h3 −0.8→+2.9). **Not adopted, see below** |

**Why #16 is not adopted (open decision for Tiffany).** The correction estimated at each training cutoff: 2025-03: +1.4%, 2025-06: +0.2%, 2025-08: +2.7%, 2025-11: +3.7% (all lower the predictions, which suited the falling 2025 market). At the final cutoff 2026-06 it is **−3.4%**, i.e. it would *raise* every June–September 2026 prediction by ~3.5%, because the February–May 2026 spring rebound makes the model look too low. The dev months never tested a correction of that sign, and June–September is the seasonal dip, so the gain on dev does not carry over with any confidence. Phase 1 model = #12.
| 17 | 2026-10-23 | `tune_a`: #12 + max_iter 300, lr 0.05, no early stopping | off | 10.36 | 13.21 | No | −0.17 vs #12 |
| 18 | 2026-10-23 | `tune_b`: #12 + max_iter 600, lr 0.05, max_leaf_nodes 63, min_samples_leaf 50, no early stopping | off | 9.70 | 12.54 | No | −0.83 vs #12; h0→h3 9.39→9.91; signed +2.5→+4.4 |
| 19 | 2026-10-23 | `tune_c`: #12 + max_iter 1500, lr 0.03, max_leaf_nodes 127, min_samples_leaf 50, no early stopping | off | **9.35** | **12.15** | No | −0.35 vs #18, −1.18 vs #12. h0→h3 9.21→9.47; signed +2.8→+4.3. Gains shrinking and each dev run now takes >10 min → stopped tuning here. **Adopted as the Phase 1 off-market model** |

## Phase 1 summary (2026-10-23)

Off-market dev MdAPE (2025-06 to 2025-11, horizons 0–3 pooled), each step on top of the previous:

| Step | Off-market MdAPE | Change |
|---|---|---|
| Pass line (community × type 12-month median) | 18.70 | — |
| Starter nearest-10 baseline | 21.00 | — |
| Starter `FeatureModel` | 15.78 | — |
| 1. Property features (area range flag, lot, garage, style, municipality) | 14.63 | −1.15 |
| 2. Listing month | 14.55 | −0.08 |
| 3. Past-only market levels (community×type, community, type, city) | 11.56 | −2.99 |
| 4. Remarks phrase groups + condition score | 11.04 | −0.52 |
| 5. Windows anchored at data end, simulated staleness, 1-year half-life weights | 10.53 | −0.51 |
| 6. Bigger trees (1500 iters, lr 0.03, 127 leaves) | **9.35** | −1.18 |

List-aware for now: property + remarks features + list price, default trees: dev MdAPE 4.71 (list-price baseline 7.09). Phase 2 will redesign it.

Final validation (2025-12 to 2026-05) not run since the Phase 0 baseline reproduction.

`python run.py predict` → both files written, `python run.py check` OK, and a second run gives byte-identical files.
Sanity: predicted medians per type sit close to the Mar–May 2026 sale medians (freehold types 2–7% below, in line with the June–September dip). List-aware predicts a median 0.955 × list price; recent quarters closed at 0.941–0.949, so it may still run ~1% high (dev bias +2%).

Open items:
- Off-market still over-predicts on dev: median signed error +2.8% (h=0) to +4.3% (h=3).
- Bias correction (#16) helps on dev but flips sign at the final cutoff (−3.4%) → needs a decision.
- Tuning stopped at #19 for run-time reasons; larger trees may still help a little.
- Remarks: only hand-picked phrases so far; TF-IDF / embeddings not tried.

## Phase 2: list-aware (branch phase2/list-aware)

| # | Date | Change | Track | Dev MdAPE | Dev MAPE | Final val touched? | Notes |
|---|---|---|---|---|---|---|---|
| 20 | 2026-10-23 | `list_price_target`: re-run of #9 list-aware (property + remarks + list_price, log price target) | list | 4.71 | 5.88 | No | Same as #9 (sanity check after refactor). signed +2.0% |
| 21 | 2026-10-23 | Step 1 `list_ratio`: same features, target log(sold/list), prediction × list price | list | 4.68 | 5.75 | No | −0.03 vs #20; MAPE −0.13. Bias unchanged (+2.1%) |
| 22 | 2026-10-23 | Step 2 `list_ratio_s2l`: + sale_to_list_features(): past-only mean log(sold/list) by community×type (3/12 mo), community (12), type (1/3/12), city (1/3/12), counts, momentum; windows anchored and staleness-simulated as in #11 | list | 4.46 | 5.54 | No | −0.22 vs #21. signed h0→h3 +1.2→+1.9. Leakage tests extended to these features |
| 23 | 2026-10-23 | Step 3 `list_ratio_s2l_vm`: + list_vs_market(): log list price minus (recent ppsf × sqft) for community×type 6/12 mo, community 12 mo, type 3 mo; log list per sqft | list | 4.32 | 5.38 | No | −0.14 vs #22. signed h0→h3 +1.0→+1.8 |
| 24 | 2026-10-23 | Step 4 `list_hl365`: #23 + recency weights (half-life 365 days) | list | **4.24** | 5.32 | No | −0.08 vs #23. signed h0→h3 +0.9→+1.6 |
| 25 | 2026-10-23 | `list_hl365_tuned`: #24 + off-market tree settings (1500 iters, lr 0.03, 127 leaves) | list | 4.35 | 5.42 | No | +0.11 worse than #24: the ratio target has less signal, big trees overfit |
| 26 | 2026-10-23 | `list_hl365_a`: #24 + 300 iters, lr 0.05, no early stopping | list | 4.24 | 5.32 | No | tie with #24 |
| 27 | 2026-10-23 | `list_hl365_b`: #24 + 600 iters, lr 0.05, 63 leaves, min leaf 50 | list | 4.29 | 5.36 | No | worse → **#24 (default trees) adopted** |

## Phase 2 summary (2026-10-23)

List-aware dev MdAPE (2025-06 to 2025-11, horizons 0–3 pooled), each step on top of the previous:

| Step | List-aware MdAPE | Change |
|---|---|---|
| List price as the prediction (starter baseline) | 7.09 | — |
| Phase 1 placeholder (#9 / #20) | 4.71 | — |
| 1. Target log(sold/list) | 4.68 | −0.03 |
| 2. Past-only sale-to-list levels | 4.46 | −0.22 |
| 3. List price vs recent neighbourhood price per sq ft | 4.32 | −0.14 |
| 4. Recency weights (half-life 1 year) | **4.24** | −0.08 |

Bigger trees did not help (#25–#27). Remaining bias: median signed +0.9% (h=0) to +1.6% (h=3).
`make_model("list_aware")` now uses #24. `predict` + `check` OK, rerun byte-identical. Predictions are a median 0.948 × list price (5–95%: 0.897–1.054); sales in 2025 Q3–2026 Q2 closed at 0.941–0.949 × list.
Final validation still not run since Phase 0.
Not tried: the full off-market valuation as a list-aware feature (needs time-respecting out-of-sample valuations for every training sale).
