# Video script (≤ 5 minutes, YouTube unlisted)

Follows the report and the 11 slides in slides_outline.md. About 700 words ≈ 4 min 50 s at a calm
pace. Numbers in 【】 are filled in after the final check; every number must come from our own
validation (experiments.md / report). Suggested split for four speakers: slides 1–4, 5–6, 7–8,
9–11.

---

## Slide 1 — Title (0:00–0:10)

Hi, we're 【team name】, and this is our entry to the SSU Data Dash: valuing Toronto homes a few
months ahead.

## Slide 2 — Table of contents (0:10–0:20)

We'll follow our report: a summary, our data and features, the models, how we validated them, the
results, where the model fails, and its limitations.

## Slide 3 — Summary (0:20–0:50)

We trained on 150,000 sales up to May 2026 and predicted homes that sold from June to September.
So our data stops up to four months before the homes sell — that gap shaped everything we did.

On our development months, the off-market model's median error is 9.19%, about half the 18.70% of
the official pass line. The list-aware model is at 4.24%, against 7.09% for just using the list
price.

## Slide 4 — Data and Features (0:50–1:35)

We used only the competition data, and gave the model three kinds of information.

First, the home itself: floor area, rooms, parking, lot size, type and location.

Second, the remarks. We turned phrases like "fully renovated" or "needs work" into condition flags
and a score. Before our development months, homes with the worst score sold about 7% below their
neighbourhood's price per square foot, and the best about 9% above.

Third, the market: what similar homes nearby sold for per square foot over the last one to twelve
months. Every one of these numbers uses only sales that closed before the home was listed, and our
unit tests check that no home ever sees its own sale or a later one.

## Slide 5 — Models (1:35–2:05)

Both tracks use gradient-boosted trees. The off-market model learns the log of the price, so a 10%
miss counts the same on a condo as on a mansion.

The list-aware model doesn't predict the price at all. It predicts the ratio of sold to list price
— the discount or premium — and multiplies back by the list price. That's a much easier thing to
learn. In both, a sale counts half as much for every year it is older, so the model follows the
recent market.

## Slide 6 — Validation (2:05–2:50)

We never split randomly: that would let the model see the future. We used walk-forward validation
— to predict a month, train only on sales before it.

Because the real test is up to four months past our data, we predicted each development month four
times, with data ending zero, one, two and three months earlier.

We made every decision on June to November 2025 — 35 logged experiments. December 2025 to May 2026
we locked away and ran exactly once, after freezing both models.

## Slide 7 — Results (2:50–3:35)

On the locked final check, run once, the off-market error was 8.98% and list-aware 4.13%, with
data ending zero to three months earlier — about the same as on our development months, and well
under both bars. That's the number we expect the hidden test to look like.

Error barely grows with distance: 9.0% when the data is fresh, 9.4% three months out.

This chart adds our ideas one at a time. Past market prices were the biggest gain, three full
points. And one idea we expected to matter — training on deliberately stale data — didn't help once
the trees were large, so we removed it.

We also give a range: on the final check, our 80% intervals covered 77.5% of sales — a little
narrow, and we say so in the report.

## Slide 8 — Where the model fails (3:35–4:10)

Condos are easiest, about 8% error. Detached, semi-detached houses and freehold townhouses are hardest, 11 to
12%: the lot shape, the renovation quality, the street aren't in the data.

Neighbourhoods with very mixed prices are harder than neighbourhoods with few sales. And for
list-aware, the misses are bidding wars and deep discounts — nothing in a listing predicts those.

## Slide 9 — Limitations (4:10–4:30)

In 2025 both models priced about 3% high, because prices kept falling; in spring 2026 that bias
vanished as prices rose. A correction helped on our development months, but at our final data date
it pointed the other way, untested — so we left it out. Our remark phrases are hand-picked, and we only tried tree models.

## Slide 10 — Outside Data (4:30–4:35)

We used no outside data.

## Slide 11 — Reproducing our results (4:35–4:50)

Four commands rebuild both submission files, byte for byte. The code, our experiment log and the
report are in our GitHub repository. Thank you.
