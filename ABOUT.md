# About this research

**Machine Learning for Demand Forecasting in Perishable Food Production Planning**
Mohammad Dauood · COM748 MSc Research Project · Supervisor: Dr Serif Salem

This file explains the project in very easy English. It covers:

1. [The project in one minute](#1-the-project-in-one-minute)
2. [Words used in this project](#2-words-used-in-this-project)
3. [The research gap](#3-the-research-gap)
4. [The research questions](#4-the-research-questions)
5. [The method, step by step](#5-the-method-step-by-step)
6. [Why these models](#6-why-these-models)
7. [How the experiments were done](#7-how-the-experiments-were-done)
8. [Main results](#8-main-results)
9. [Limitations](#9-limitations)
10. [Questions people may ask, with short answers](#10-questions-people-may-ask-with-short-answers)

---

## 1. The project in one minute

A meal delivery company must buy fresh food every week, **before** it knows how many people will order.

- If it buys **too much**, food goes bad and is thrown away (**waste**).
- If it buys **too little**, food runs out and customers cannot get their meal (**shortage**).

Most studies only ask: *"How close is the forecast?"* They treat 10 meals too many and 10 meals too few as
the same mistake. But in real life these two mistakes often cost **different** amounts.

This project:

1. forecasts weekly meal orders 1 to 10 weeks ahead,
2. compares simple methods, normal machine learning models and **quantile** models,
3. turns each forecast into an order, and estimates how much **waste, shortage and cost** it would cause.

**Main message:** the most accurate model is not always the cheapest one. When a missed order costs more than
a wasted meal, it is better to order a "safer" amount, and quantile models do this well.

---

## 2. Words used in this project

| Word | Simple meaning |
|---|---|
| **Forecast** | A guess of a future number, for example "next week meal 1885 at centre 55 will get 180 orders". |
| **Demand / orders** | How many meals people actually ordered in a week. |
| **Perishable food** | Fresh food that goes bad quickly, so it cannot be kept for next week. |
| **Horizon (weeks ahead)** | How far into the future we forecast. 1 = next week, 10 = ten weeks from now. |
| **Series** | One meal at one centre over time. There are 3,597 of them. |
| **Feature / input** | Information the model uses, for example last week's orders or the price. |
| **Data leakage** | Using future information by mistake, like seeing the exam answers before the exam. It makes results look better than they really are. |
| **Baseline** | A very simple method that a good model must beat, for example "next week = last week". |
| **Naive forecast** | The simplest baseline: copy last week's orders. |
| **Moving average** | The average of the last 26 weeks, used as the forecast. |
| **Linear Regression** | A simple model that draws a straight-line relationship between the inputs and the orders. |
| **Tree model** | A model made of many yes/no questions, for example "Is there an offer? Is it a drink?" |
| **Random Forest** | Many trees built separately, and their answers are averaged. |
| **XGBoost / LightGBM** | Trees built one after another, where each new tree fixes the mistakes of the earlier ones ("boosting"). Very strong on table data. |
| **Point forecast** | One best-guess number (usually aims at the average). |
| **Quantile** | A level of demand. The 0.8 quantile is a number that demand stays below in about 8 out of 10 weeks. |
| **Quantile model** | A model that forecasts a chosen quantile instead of the average, so it gives a "safer" (or lower) number on purpose. |
| **Quantile level (τ, "tau")** | Which quantile we want, from 0.3 to 0.9. Higher level = bigger, safer order. |
| **Log scale** | A maths tool that makes big numbers much smaller but keeps small numbers almost the same. Example: 10 → 2.4, 100 → 4.6, 1,000 → 6.9, 20,000 → 9.9. It stops a few huge weeks from confusing the model. At the end we change the answer back into meals. |
| **Rolling-origin testing** | Train on the past, test on the next weeks, then move forward and do it again. Like a real planner, the model never sees the future. |
| **Fold / round** | One train-then-test step. We have 3 rounds. |
| **Hyperparameter tuning** | Choosing model settings (like tree depth). We do this on training weeks only. |
| **Early stopping** | Stop adding trees when the model stops getting better on the check weeks. Stops it learning noise. |
| **RMSE** | Average error in meals, where big mistakes count extra. Lower is better. |
| **MAE** | Average error in meals. Lower is better. |
| **RMSLE** | Average error on the log scale, so it looks at errors in relative (percent-like) terms. Lower is better. |
| **WAPE** | Total error divided by total orders, in percent. Example: real orders 1,000, total error 320 → 32%. Lower is better. |
| **Bias** | Is the forecast too high or too low on average? Negative = too low. |
| **Pinball loss** | The score for quantile forecasts. It punishes a forecast that is too low more when the level is high (and the other way round). Lower is better. |
| **Coverage** | How often real orders were below the forecast. For a 0.8 forecast, it should be about 80%. |
| **Newsvendor rule** | An old rule from stock planning: the more a missed sale costs compared with a wasted item, the higher you should order. |
| **Cost ratio** | How many times more a missed order costs than a wasted meal. Example: ratio 3 = a missed order costs 3 times as much. |
| **Best level (τ\*)** | The quantile level the newsvendor rule says is cheapest for a given cost ratio. Ratio 1 → 0.5, ratio 3 → 0.75, ratio 5 → 0.83. |
| **Safety stock** | A little extra added on top of the normal forecast, as real planners often do. |
| **Simulation** | Estimating waste, shortage and cost with clear rules, because the data has no real cost numbers. |
| **Service level** | How often there was no shortage at all. |
| **Diebold–Mariano (DM) test** | A test that checks if one model is **really** better than another, or if the difference is just luck. p below 0.05 = really better. |
| **Corrected DM test** | The same test, adjusted because errors in nearby weeks are linked and we only have 30 test weeks. |
| **Bootstrap** | Re-sample the data many times (here 1,000 times) to get a range for a result, for example "a saving between 24% and 26%". |
| **95% confidence interval** | The range where we are fairly sure the true value is. |
| **CRISP-DM** | A common step-by-step plan for data science projects: understand the problem, understand the data, prepare the data, build models, test them, use them. |

---

## 3. The research gap

What other studies did, and what is missing:

| Study | Data | Looked at waste or cost? |
|---|---|---|
| Panda & Mohanty (2023) | Genpact meals (public, same as ours) | No, only how close the forecast is |
| Rodrigues et al. (2024) | Catering (private) | Yes |
| Malefors et al. (2022) | School catering (private) | Yes |
| Garre et al. (2020) | Food company (private) | Yes |
| M5 Uncertainty contest (2022) | Walmart shops (public) | Quantiles only, not food waste |
| **This study** | **Genpact meals (public)** | **Yes, estimated** |

So there are three gaps:

1. **No quantile models on this public food data.** The only study on it (Panda & Mohanty) used normal
   models and normal error scores only.
2. **Waste studies cannot be repeated.** The studies that link forecasts to food waste all use private data.
3. **No fair, public study links forecasts to waste, shortage and cost**, with testing that avoids data leakage.

This project fills these gaps with public data, open code and fair (time-based) testing.

---

## 4. The research questions

| | Question in easy words | Type of question |
|---|---|---|
| **RQ1** | Are tree models (Random Forest, XGBoost, LightGBM) more accurate than simple methods? | Accuracy / comparison |
| **RQ2** | Do quantile models change how often we forecast too high or too low, and are their levels right? | Error direction / reliability |
| **RQ3** | When the cost of waste and shortage changes, which model and level give the lowest cost? Does it match the newsvendor rule? | Decision / cost / sensitivity |

---

## 5. The method, step by step

The project follows **CRISP-DM**, with one extra step for the cost estimate. Each step has its own notebook.

| Step | What happens | Notebook |
|---|---|---|
| 1. Get the data | Download the public Genpact data (456,548 rows: 51 meals, 77 centres, 145 weeks). Check the files have not changed. | 01 |
| 2. Understand the data | Charts and checks: no empty values, no repeated rows, no zero weeks. Orders go from 13 to 24,299 (middle value 136). Offers almost triple the orders. | 02 |
| 3. Prepare the data | Join the 3 files. Build 25 inputs from past weeks only. Learn on the log scale. Make the 3 train/test rounds. | 03 |
| 4. Simple models | Copy last week, 26-week average, Linear Regression. | 04 |
| 5. Tree models | Random Forest, XGBoost, LightGBM. | 05 |
| 6. Quantile models | Linear Quantile Regression, quantile XGBoost, quantile LightGBM, for levels 0.3 to 0.9. | 06 |
| 7. Cost estimate | Turn forecasts into orders, count waste and shortage, and add up the cost for 7 cost ratios. | 07 |
| 8. Extra checks | Fairer comparison with a safety stock, corrected test, and ranges (bootstrap). | 08 |
| 9. App | A small Streamlit app where a planner can try different cost ratios. | app/ |

### The 25 inputs

- **Weeks ahead** (1 to 10), so one model can forecast all horizons.
- **Past orders:** last week, last known week, weeks since last seen, averages over 4, 8 and 26 weeks and over all
  past weeks, how much orders change (spread over 8 weeks), how many weeks of history.
- **That week's price and offers:** checkout price, base price, discount, email offer, home-page offer, and how
  the price compares with the usual price. These are planned in advance, so they are known.
- **Centre and meal details:** centre, meal, city, region, centre type, centre size, meal type, cuisine.

### Important choices

- **One model for all series and all horizons** ("global model"), which gives 4.4 million training examples.
- **Missing weeks** (65,017) are not filled with zero, because they mean the meal was not on sale.
- **Very large values** (7.2%) are kept, because they are real orders from big centres and offers.
- **The competition test file is not used**, because it has no real order numbers.

---

## 6. Why these models

| Group | Models | Why |
|---|---|---|
| Simple | Copy last week, 26-week average, Linear Regression | A fair starting point. Any machine learning model must beat these to be useful. |
| Normal ML | Random Forest, XGBoost, LightGBM | Tree models are the strongest on table data like this. LightGBM won the big M5 forecasting contest. They handle many kinds of inputs and are fast on millions of rows. |
| Quantile | Linear Quantile Regression, quantile XGBoost, quantile LightGBM | They give a chosen level, not just the average, which is what the newsvendor rule needs. Quantile LightGBM won the M5 Uncertainty contest. Linear QR is the simple quantile baseline. |

**Why not deep learning (LSTM)?** Research shows tree models still beat deep learning on table data. Deep learning
is also much slower and harder to tune, and the project had 12 weeks. It is listed as future work.

**Why levels 0.3 to 0.9?** Each level matches a cost ratio through the newsvendor rule. Ratio 0.5 → 0.33,
ratio 1 → 0.5, ratio 2 → 0.67, ratio 3 → 0.75, ratio 5 → 0.83. Steps of 0.1 cover all of these with only 7 models.

---

## 7. How the experiments were done

### Testing plan (rolling-origin, 3 rounds)

| Round | Train on weeks | Test on weeks |
|---|---|---|
| 1 | 1 – 115 | 116 – 125 |
| 2 | 1 – 125 | 126 – 135 |
| 3 | 1 – 135 | 136 – 145 |

- In each round, the model is only trained on weeks **before** the test weeks. That gives 30 test weeks in total.
- Settings are chosen using the **last 10 training weeks** as check weeks, never the test weeks.
- An automatic test checks that no input uses future weeks.

### Training details

- All models learn on the log scale; forecasts are changed back into meals.
- XGBoost and LightGBM use early stopping (stop when the check weeks stop improving).
- Random Forest is trained on a random 1 million rows, because it is slow on 4 million rows.
- Linear Quantile Regression is trained on a random 300,000 rows for the same reason.
- Quantile LightGBM trains one model per level. If a lower level gives a bigger number than a higher level
  ("crossing"), the numbers are sorted. Quantile XGBoost gives all 7 levels from one model.

### Cost estimate rules

1. We make exactly what the forecast says (rounded to whole meals).
2. Meals not sold that week are thrown away (waste).
3. Orders we cannot serve are lost (shortage).
4. Recorded orders are treated as the true demand.
5. A wasted meal costs 1. A missed order costs 0.5, 1, 1.5, 2, 3, 4 or 5 (the cost ratio), because real prices are unknown.

Example: we make 100 meals. If 80 are ordered, 20 are wasted. If 120 are ordered, 20 are missed.
With ratio 3, the 20 missed orders cost 60, but the 20 wasted meals cost only 20.

For each ratio, quantile models order at the level the newsvendor rule gives. This level comes from the rule,
**not** from looking at test weeks, so there is no leakage.

### Extra checks (notebook 08)

- **Safety stock baseline:** the normal LightGBM forecast plus a little extra, learned from its own past errors in
  the 10 weeks before the test period. This is a fairer and stronger rival than the plain forecast.
- **Corrected Diebold–Mariano test** for every claim.
- **Bootstrap:** 1,000 re-samples of the 3,597 series give a 95% range for each saving.

---

## 8. Main results

### RQ1 – Accuracy (mean of 3 rounds)

| Model | WAPE | RMSE | MAE | RMSLE |
|---|---|---|---|---|
| Copy last week | 50.9% | 294 | 127 | 0.81 |
| 26-week average | 38.4% | 237 | 96 | 0.62 |
| Linear Regression | 32.0% | 183 | 80 | 0.51 |
| Random Forest | 29.4% | 164 | 73 | 0.49 |
| XGBoost | 28.7% | 152 | 71 | 0.49 |
| **LightGBM** | **28.6%** | **152** | **71** | **0.49** |

- All three tree models beat every simple method in every round. The corrected DM test says this is real (p < 0.001).
- LightGBM and XGBoost are a tie (p = 0.96). LightGBM also beats Random Forest (p = 0.002).
- All models forecast **too low** on average: only about 46% of forecasts are above the real orders.

### RQ2 – Quantile models

- The levels are close to what they promise. For level 0.8, real orders were below the forecast in about 77–80%
  of cases (quantile LightGBM 77%, quantile XGBoost 78%, Linear QR 80%).
- The normal LightGBM forecast was above the real orders only 46% of the time, so it cannot be used as a "safe" order.
- Quantile LightGBM has a lower pinball loss than the normal forecast at every level. The corrected DM test
  confirms this at levels 0.5, 0.7 and 0.9 (p < 0.001).

### RQ3 – Cost (normal LightGBM forecast = 100, lower is cheaper)

| Cost ratio | Best level | Normal Linear Regression | Safety stock | Quantile XGBoost | **Quantile LightGBM** |
|---|---|---|---|---|---|
| 0.5 | 0.33 | 109.7 | 99.8 | 96.0 | **95.7** |
| 1 | 0.50 | 111.6 | 101.5 | 99.0 | **98.1** |
| 2 | 0.67 | 113.2 | 89.3 | 86.2 | **86.2** |
| 3 | 0.75 | 113.8 | 78.5 | 74.5 | **75.0** |
| 5 | 0.83 | 114.5 | 64.0 | 59.0 | **59.6** |

- Quantile LightGBM saves **2% to 40%** compared with the normal forecast. The more a missed order costs, the bigger the saving.
- It also beats the fairer **safety stock** rival by **3% to 7%** at every cost ratio, and the corrected DM test
  says this is real (p below 0.05 for all ratios).
- The bootstrap ranges are narrow, for example a saving of 25.0% (range 24.1% to 25.8%) at ratio 3.
- The cheapest level found in the data is close to the newsvendor rule (for example 0.5 at ratio 1, 0.8 at ratio 4).

### Compared with Panda & Mohanty (same data, same test weeks 136–145)

| | RMSE | MAE |
|---|---|---|
| Their best tree model (CatBoost) | 186.7 | 99.4 |
| Our LightGBM | 105.7 | 62.4 |

Our tree models have about **40% lower** RMSE and MAE. Their RMSLE is lower (0.29 vs our 0.48), but their exact
inputs and steps are not fully clear, so this number is hard to compare directly.

---

## 9. Limitations

1. **Costs are estimated, not real.** The data has no prices for waste or shortage, so we test a range of cost ratios.
   The savings are estimates, not real money.
2. **One dataset.** Only the Genpact meal data is used. Results may be different for other companies.
3. **Simple rules.** Each week is treated on its own: food cannot be kept for the next week, and a missed customer
   does not come back later. Real kitchens may have more complex rules.
4. **Recorded orders = true demand.** If food ran out, some demand may not have been recorded (hidden demand).
5. **Missing weeks are skipped.** We assume a missing week means the meal was not on sale.
6. **Only 30 test weeks.** That is why we used the corrected test and the bootstrap.
7. **No deep learning.** LSTM and similar models were not tested (time limit).
8. **Data is old and anonymous.** We do not know the real company, city names or dates, so we cannot add
   holidays or weather.

---

## 10. Questions people may ask, with short answers

### About the problem and the gap

**Q: Why is this problem important?**
A: About 1.05 billion tonnes of food were wasted in 2022 (UNEP), and food service caused about 28%. Better orders
mean less waste and fewer unhappy customers.

**Q: What is new in your work?**
A: Three things: quantile models on this public food data for the first time; turning forecasts into waste,
shortage and cost; and fair, time-based testing that others can repeat with open data and code.

**Q: How is your work different from Panda & Mohanty?**
A: They only checked accuracy. I also check the direction of errors and the cost of waste and shortage, use
quantile models, and use rolling-origin testing in three rounds instead of one split.

### About the data

**Q: Why this dataset?**
A: It is a real weekly planning problem, it is public, it has no personal data, and another study used it, so I can compare.

**Q: Why did you not fill missing weeks with zero?**
A: The smallest recorded value is 13 and there are no zeros, so a missing week most likely means the meal was not on sale.

**Q: Why did you keep the very large values?**
A: They come from big centres and special offers, so they are real orders. Removing them would make the model
worse in exactly the weeks that matter most.

**Q: Why the log scale?**
A: Orders go from 13 to over 24,000. The log scale makes big numbers smaller, so a few huge weeks do not take over
the model. The answers are changed back into meals at the end.

**Q: Is it fair to use the price and offer of the target week?**
A: Yes. Prices and offers are planned in advance by the company, so they are known when the order is made.

### About the method

**Q: How do you make sure there is no data leakage?**
A: Every input uses only weeks before the forecast week. Settings are chosen on training weeks only. The best quantile
level comes from the newsvendor rule, not from test weeks. An automatic test checks the inputs.

**Q: Why rolling-origin testing and not a random split?**
A: A random split lets the model learn from future weeks, which is not possible in real life. Rolling-origin testing
copies how a planner really works: only the past is known.

**Q: Why 1 to 10 weeks ahead?**
A: Weeks 1–2 help to buy ingredients, weeks 3–10 help to plan with suppliers. Panda & Mohanty also used 10 weeks.

**Q: Why one global model and not one model per meal?**
A: 3,597 small models would each have very little data. One global model learns from all series together, which
works better for tree models (Januschowski et al., 2022) and is much easier to manage.

**Q: How did you choose the model settings?**
A: In each round, I tried a few settings on the last 10 training weeks and kept the best. XGBoost and LightGBM also
use early stopping.

### About the models

**Q: Why tree models?**
A: They are the strongest on table data, they handle many kinds of inputs, they are fast, and LightGBM won the M5 contest.

**Q: Why not LSTM or other deep learning?**
A: Research shows tree models still beat deep learning on table data, deep learning is slower and harder to tune, and
I had 12 weeks. It is future work.

**Q: What is a quantile model, in simple words?**
A: A normal model gives the average guess. A quantile model gives a number that is enough in, say, 8 out of 10 weeks.
You choose how safe you want to be.

**Q: Why not just add a safety stock to the normal forecast?**
A: I tested that too (notebook 08). Quantile LightGBM was still 3% to 7% cheaper at every cost ratio, and the
difference is real (corrected DM test).

**Q: What is "crossing" and how did you fix it?**
A: Sometimes the 0.7 forecast is bigger than the 0.8 forecast, which makes no sense. For quantile LightGBM I sort the
7 numbers so they always go up. Quantile XGBoost gives all levels from one model, so it had no crossing.

### About the results

**Q: Which model is best?**
A: For accuracy, LightGBM and XGBoost (tie, WAPE about 28.6%). For cost, quantile LightGBM or quantile XGBoost at
the level given by the newsvendor rule.

**Q: Is the difference real or luck?**
A: I used the Diebold–Mariano test, corrected for linked weeks and few test weeks, plus a bootstrap. The main
differences are real (p < 0.05), with narrow ranges.

**Q: Why are all models too low on average?**
A: Most weeks are small but a few are huge (offers, big centres). A model that aims for the average on the log scale
lands near the typical week, so it misses the big ones. That is exactly why quantile models help.

**Q: What does a WAPE of 28.6% mean in practice?**
A: Across all meals and centres, the total forecast error is about 29% of the total orders. That is about 10% better
than Linear Regression (32%).

**Q: Your RMSLE is higher than Panda & Mohanty's. Why?**
A: Our RMSE and MAE are about 40% lower on the same test weeks. Their steps (inputs, how they built past values) are
not fully described, and some of their numbers do not match each other, so RMSLE is hard to compare directly.

**Q: Is the saving real money?**
A: No. It is an estimate under clear rules and a range of cost ratios. The pattern (higher cost of shortage →
bigger saving) is the main finding, not the exact percent.

**Q: What should a company do with this?**
A: First estimate how much a missed order costs compared with a wasted meal. Then order at the matching quantile
level, using a quantile model. The Streamlit app shows this.

### About limitations and future work

**Q: What is the biggest weakness?**
A: The costs are not real; they are estimated over a range. Also only one dataset is used.

**Q: Would results hold for another company?**
A: The method can be used anywhere, but the exact numbers may change. Testing on more datasets is future work.

**Q: What would you do next?**
A: Use real cost data from a company; test more datasets; try deep learning; allow food to be kept for more than
one week (multi-week stock); and handle hidden demand when food runs out.

**Q: Can I repeat your work?**
A: Yes. The data is public, all code is in this repository, each step has its own notebook, and the README explains
how to run everything.

---

*All numbers come from the tables in `results/tables/`. Full references are in the project report.*
