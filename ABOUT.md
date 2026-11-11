# About this research

**Machine Learning for Demand Forecasting in Perishable Food Production Planning**
Mohammad Dauood · COM748 MSc Research Project · Supervisor: Dr Serif Salem

This file explains the project in very simple English, with an easy example for each idea.

1. [The project in one minute](#1-the-project-in-one-minute)
2. [Words used in this project](#2-words-used-in-this-project)
3. [What is missing in other studies (the research gap)](#3-what-is-missing-in-other-studies-the-research-gap)
4. [My three research questions](#4-my-three-research-questions)
5. [My method, step by step](#5-my-method-step-by-step)
6. [Why I chose these models](#6-why-i-chose-these-models)
7. [How I did the experiments](#7-how-i-did-the-experiments)
8. [Main results](#8-main-results)
9. [Limitations (what my study cannot show)](#9-limitations-what-my-study-cannot-show)
10. [Questions people may ask, with short answers](#10-questions-people-may-ask-with-short-answers)

---

## 1. The project in one minute

A meal delivery company must buy fresh food every week. But it buys the food **before** it knows how many people
will order.

> **Example:** A kitchen thinks 100 people will order chicken curry next week, so it makes 100 meals.
> - If only 80 people order, **20 meals go in the bin**. This is **waste**.
> - If 120 people order, **20 people get nothing**. This is **shortage**.

Most studies only ask: *"How close was the forecast?"* For them, 20 too many and 20 too few are the **same**
mistake. But in real life, one mistake can cost much more than the other.

> **Example:** A wasted meal costs the kitchen £3 of food. A missed order loses the £12 sale and makes the customer
> unhappy. So here a missed order costs **4 times** more than a wasted meal.

My project does three things:

1. It forecasts how many meals people will order, 1 to 10 weeks ahead.
2. It compares simple methods, normal machine learning models and **quantile** models (models that give a "safer" number).
3. It turns each forecast into an order, and counts the **waste, shortage and cost**.

**Main message:** the most accurate model is not always the cheapest. When a missed order costs more than a
wasted meal, it is better to make a bit more, and quantile models do this well.

---

## 2. Words used in this project

| Word | Simple meaning | Easy example |
|---|---|---|
| **Forecast** | A guess of a future number. | "Next week, meal 1885 at centre 55 will get 180 orders." |
| **Demand / orders** | How many meals people really ordered. | In week 1, centre 55 sold 177 of meal 1885. |
| **Perishable food** | Fresh food that goes bad quickly. | Salad made on Monday cannot be sold next Monday. |
| **Weeks ahead (horizon)** | How far into the future we forecast. | Standing in week 135, "1 week ahead" is week 136, "10 weeks ahead" is week 145. |
| **Series** | One meal at one centre, week after week. | Meal 1885 at centre 55, from week 1 to week 145. There are 3,597 of these. |
| **Input (feature)** | Information the model looks at. | Last week's orders, the price, and whether there is an offer. |
| **Data leakage** | Using future information by mistake. | Like seeing the exam answers before the exam. Your score looks great, but it is not real. |
| **Baseline** | A very simple method that a good model must beat. | "Next week will be the same as last week." |
| **Naive forecast** | Copy last week. | Last week 150 orders, so forecast 150. |
| **Moving average** | The average of the last 26 weeks. | If the last 26 weeks average 140, forecast 140. |
| **Linear Regression** | A simple model that adds up the inputs with weights. | "Orders = past average + a bit more if there is an offer − a bit less if the price is high." |
| **Tree model** | A model that asks many yes/no questions. | "Is there an offer? Yes. Is it a drink? Yes. Then expect about 400 orders." |
| **Random Forest** | Many trees, each gives an answer, and we take the average. | 100 friends each guess the orders, and we use the average guess. |
| **XGBoost / LightGBM** | Trees built one after another; each new tree fixes the mistakes of the last ones. | A student who checks the last test, sees the mistakes, and studies only those. |
| **Point forecast** | One best guess, usually the "middle" value. | "About 100 orders." |
| **Quantile** | A level that demand stays below most of the time. | The 0.8 quantile is a number that is enough in about 8 out of 10 weeks. |
| **Quantile model** | A model that forecasts a chosen level instead of the middle. | Normal model: "100 meals". Quantile model at 0.8: "130 meals, enough in 8 of 10 weeks". |
| **Quantile level (τ, "tau")** | How safe we want to be, from 0.3 to 0.9. | 0.5 = middle. 0.9 = very safe, big order. 0.3 = small order. |
| **Log scale** | A maths tool that makes big numbers much smaller, but keeps small numbers almost the same. | 10 → 2.4, 100 → 4.6, 1,000 → 6.9, 20,000 → 9.9. Now 20,000 is only about twice 100, not 200 times. At the end we change the answer back into meals. |
| **Rolling-origin testing** | Train on the past, test on the next weeks, then move forward and do it again. | Learn from weeks 1–115, test on 116–125. Then learn from 1–125, test on 126–135. |
| **Round (fold)** | One "train, then test" step. | We do 3 rounds. |
| **Tuning** | Choosing the best settings for a model. | Trying trees with 8 levels and 16 levels, and keeping the better one. |
| **Early stopping** | Stop adding trees when the model stops getting better. | Like stopping study when your practice score no longer goes up. |
| **Error** | How far the forecast is from the real orders. | Forecast 90, real 100, so the error is 10 meals. |
| **MAE** | The average error in meals. | Errors of 10, 20 and 30 meals give MAE = 20. |
| **RMSE** | Like MAE, but big mistakes count extra. | One error of 100 hurts much more than ten errors of 10. |
| **RMSLE** | The error on the log scale, so it looks at errors in percent-like terms. | Missing by 10 on a week of 20 counts more than missing by 10 on a week of 1,000. |
| **WAPE** | Total error divided by total orders, in percent. | Total orders 1,000, total error 320, so WAPE = 32%. |
| **Bias** | Is the forecast too high or too low on average? | Bias −20 means the forecast is 20 meals too low on average. |
| **Pinball loss** | The score for quantile forecasts. Lower is better. | For a 0.8 forecast, being too low is punished 4 times more than being too high. |
| **Coverage** | How often real orders were below the forecast. | A 0.8 forecast should be "enough" in about 80 of 100 weeks. |
| **Newsvendor rule** | An old rule: the more a missed sale costs, the more you should stock. | A newspaper seller who earns a lot per paper but loses little on unsold ones should keep extra papers. |
| **Cost ratio** | How many times more a missed order costs than a wasted meal. | Ratio 3: a missed order costs £3 for every £1 of a wasted meal. |
| **Best level (τ\*)** | The level the newsvendor rule says is cheapest. | Ratio 1 → order at 0.5 (the middle). Ratio 3 → 0.75. Ratio 5 → 0.83. |
| **Safety stock** | A small extra amount added on top of the normal forecast. | Forecast 100, add 15 extra "just in case", so order 115. |
| **Simulation** | Estimating waste and cost with simple rules, because there is no real cost data. | "If we made 100 and 80 were ordered, 20 were wasted." |
| **Service level** | How often there was no shortage at all. | 80% service level = no one went without a meal in 80 of 100 weeks. |
| **Diebold–Mariano (DM) test** | Checks if one model is **really** better, or just lucky. | Like checking if a coin is really unfair, or you just got 6 heads by chance. p below 0.05 = really better. |
| **Corrected DM test** | The same test, made safer because nearby weeks are linked and we only have 30 test weeks. | A busy week is often followed by another busy week, so the weeks are not fully separate. |
| **Bootstrap** | Pick the data again at random many times, and see how much the result changes. | Pick 3,597 series with repeats, 1,000 times. If the saving is always 24–26%, we can trust it. |
| **95% range (confidence interval)** | The range where we are fairly sure the true value is. | "The saving is 25%, and we are fairly sure it is between 24% and 26%." |
| **CRISP-DM** | A common step-by-step plan for data projects. | Understand the problem → understand the data → clean it → build models → test them → use them. |

---

## 3. What is missing in other studies (the research gap)

| Study | Data | Did they look at waste or cost? |
|---|---|---|
| Panda & Mohanty (2023) | Genpact meals (public, the same as mine) | No, only how close the forecast was |
| Rodrigues et al. (2024) | Catering (private) | Yes |
| Malefors et al. (2022) | School catering (private) | Yes |
| Garre et al. (2020) | Food company (private) | Yes |
| M5 Uncertainty contest (2022) | Walmart shops (public) | Used quantiles, but not for food waste |
| **My study** | **Genpact meals (public)** | **Yes, estimated** |

There are three gaps:

1. **No quantile models on this public food data.** The only study on it used normal models and only checked accuracy.
2. **Nobody can repeat the waste studies.** They all use private company data.
   *Example:* if a study uses a school's private records, no other researcher can check the results.
3. **No fair, public study links forecasts to waste, shortage and cost.**

My project fills these gaps. It uses public data and open code, and it tests the models fairly over time.

---

## 4. My three research questions

| | Question in simple words | Easy example of what it means |
|---|---|---|
| **RQ1** | Are tree models more accurate than simple methods? | Does LightGBM make smaller mistakes than "copy last week"? |
| **RQ2** | Do quantile models change how often we forecast too high or too low? And are their levels honest? | Is a "0.8" forecast really enough in about 8 of 10 weeks? |
| **RQ3** | When costs change, which model and level give the lowest cost? | If a missed order costs 3 times a wasted meal, what should the kitchen order? |

---

## 5. My method, step by step

I follow **CRISP-DM** (a common step-by-step plan), with one extra step for the cost estimate.
Each step has its own notebook.

| Step | What I did | Notebook |
|---|---|---|
| 1. Get the data | Download the public Genpact data: 456,548 rows, 51 meals, 77 centres, 145 weeks. | 01 |
| 2. Understand the data | Make charts and check the data. No empty values, no repeated rows. Orders go from 13 to 24,299 a week. Offers almost triple the orders. | 02 |
| 3. Prepare the data | Join the 3 files. Build 25 inputs from past weeks only. Use the log scale. Make the 3 test rounds. | 03 |
| 4. Simple models | Copy last week, 26-week average, Linear Regression. | 04 |
| 5. Tree models | Random Forest, XGBoost, LightGBM. | 05 |
| 6. Quantile models | Linear Quantile Regression, quantile XGBoost, quantile LightGBM, at levels 0.3 to 0.9. | 06 |
| 7. Cost estimate | Turn forecasts into orders, count waste and shortage, and add up the cost. | 07 |
| 8. Extra checks | Compare with a safety stock, use the corrected test, and work out the ranges. | 08 |
| 9. App | A small web app where a planner can try different costs. | app/ |

### The 25 inputs, with an example

To forecast **meal 1885 at centre 55, three weeks ahead**, the model looks at:

- **Weeks ahead:** 3.
- **Past orders:** last week's orders, the averages of the last 4, 8 and 26 weeks, the average of all past
  weeks, how much the orders jump up and down, and how many weeks of history there are.
- **Price and offers in that week:** the price, the discount, and whether there is an email or home-page offer.
  The company plans these in advance, so they are known.
- **Centre and meal details:** which centre and meal, city, region, centre type and size, meal type and cuisine.

### Some important choices

- **One model for everything.** One model learns from all 3,597 series and all 10 weeks ahead
  (4.4 million examples). *Example:* a small meal can learn from the patterns of similar big meals.
- **Missing weeks are not filled with zero.** 65,017 weeks have no record. That means the meal was not on sale,
  not that nobody ordered it.
- **Very large weeks are kept.** 7.2% of weeks are very large, but they are real (big centres, special offers).
- **The competition's test file is not used**, because it has no real order numbers.

---

## 6. Why I chose these models

| Group | Models | Why |
|---|---|---|
| Simple | Copy last week, 26-week average, Linear Regression | A fair starting point. If a smart model cannot beat "copy last week", it is not useful. |
| Normal ML | Random Forest, XGBoost, LightGBM | Tree models are the strongest on table data like mine. LightGBM won the big M5 forecasting contest. They are fast even on millions of rows. |
| Quantile | Linear Quantile Regression, quantile XGBoost, quantile LightGBM | They give a chosen safety level, not just the middle. This is exactly what the newsvendor rule needs. Quantile LightGBM won the M5 Uncertainty contest. |

**Why not deep learning (LSTM)?** Research shows tree models still beat deep learning on table data.
Deep learning is also slower and harder to set up, and I had 12 weeks. It is future work.

**Why levels 0.3 to 0.9?** Each level matches a cost ratio through the newsvendor rule:

| If a missed order costs… | 0.5× a wasted meal | 1× | 2× | 3× | 4× | 5× |
|---|---|---|---|---|---|---|
| …the best level is | 0.33 | 0.50 | 0.67 | 0.75 | 0.80 | 0.83 |

So 0.3 to 0.9, in steps of 0.1, covers all of these with only 7 models.

---

## 7. How I did the experiments

### Testing plan: train on the past, test on the future (3 rounds)

| Round | Learn from weeks | Test on weeks |
|---|---|---|
| 1 | 1 – 115 | 116 – 125 |
| 2 | 1 – 125 | 126 – 135 |
| 3 | 1 – 135 | 136 – 145 |

- The model **never** sees the weeks it is tested on. This gives 30 test weeks in total.
- I choose the settings using the last 10 learning weeks, never the test weeks.
- An automatic test checks that no input uses future weeks.

*Example:* in round 3, the model learns from weeks 1 to 135, exactly like a planner standing in week 135.
Then it forecasts weeks 136 to 145, and only then do we compare with what really happened.

### Training details

- All models learn on the log scale, and the answers are changed back into meals.
- XGBoost and LightGBM stop adding trees when they stop getting better (early stopping).
- Random Forest learns from 1 million random rows, because it is slow on 4 million.
- Linear Quantile Regression learns from 300,000 random rows, for the same reason.
- Quantile LightGBM makes one model per level. Sometimes the 0.7 answer is bigger than the 0.8 answer, which makes
  no sense, so I sort the answers into order. Quantile XGBoost gives all 7 levels from one model.

### The cost rules

1. The kitchen makes exactly what the forecast says (in whole meals).
2. Meals not sold that week are thrown away.
3. Orders the kitchen cannot serve are lost.
4. The recorded orders are the true demand.
5. A wasted meal costs 1. A missed order costs 0.5, 1, 1.5, 2, 3, 4 or 5 (the cost ratio), because real prices are unknown.

> **Example (cost ratio 3):** the kitchen makes 100 meals.
> - 80 are ordered → 20 wasted → cost 20 × 1 = **20**.
> - 120 are ordered → 20 missed → cost 20 × 3 = **60**.
>
> So with ratio 3, running out is 3 times worse, and it is smarter to make a bit more.

For each cost ratio, the quantile models order at the level the newsvendor rule gives. This level comes from the
rule, **not** from looking at the test weeks, so there is no cheating.

### Extra checks (notebook 08)

- **Safety stock rival.** Real planners often add a little extra to the normal forecast. I built this too: the
  normal LightGBM forecast plus an extra amount learned from its own past mistakes (from the 10 weeks before the test).
  *Example:* if the model was usually 15% too low in the past, add about 15%.
- **Corrected DM test** for every claim.
- **Bootstrap:** 1,000 random re-picks of the 3,597 series give a range for each saving.

---

## 8. Main results

### RQ1 – Which model is most accurate? (average of 3 rounds, lower is better)

| Model | WAPE | RMSE | MAE | RMSLE |
|---|---|---|---|---|
| Copy last week | 50.9% | 294 | 127 | 0.81 |
| 26-week average | 38.4% | 237 | 96 | 0.62 |
| Linear Regression | 32.0% | 183 | 80 | 0.51 |
| Random Forest | 29.4% | 164 | 73 | 0.49 |
| XGBoost | 28.7% | 152 | 71 | 0.49 |
| **LightGBM** | **28.6%** | **152** | **71** | **0.49** |

- **Answer: yes.** All three tree models beat every simple method in every round. The corrected DM test says this
  is real, not luck (p < 0.001).
- LightGBM and XGBoost are a tie (p = 0.96). LightGBM also beats Random Forest (p = 0.002).
- *Example:* if the total real orders are 1,000, LightGBM is wrong by about 286 in total, and Linear Regression by about 320.
- All models forecast **too low** on average. Only about 46% of forecasts were above the real orders.

### RQ2 – Are the quantile models honest?

- **Answer: yes.** A 0.8 forecast was enough in about 77–80% of cases (quantile LightGBM 77%, quantile XGBoost 78%,
  Linear QR 80%). That is close to the promised 80%.
- The normal LightGBM forecast was enough only 46% of the time, so it is not a "safe" order.
- Quantile LightGBM has a lower pinball loss than the normal forecast at every level. The corrected DM test
  confirms this at levels 0.5, 0.7 and 0.9 (p < 0.001).

### RQ3 – Which order is cheapest? (normal LightGBM forecast = 100, lower is cheaper)

| A missed order costs… | Best level | Normal Linear Regression | Safety stock | Quantile XGBoost | **Quantile LightGBM** |
|---|---|---|---|---|---|
| 0.5× a wasted meal | 0.33 | 109.7 | 99.8 | 96.0 | **95.7** |
| 1× | 0.50 | 111.6 | 101.5 | 99.0 | **98.1** |
| 2× | 0.67 | 113.2 | 89.3 | 86.2 | **86.2** |
| 3× | 0.75 | 113.8 | 78.5 | 74.5 | **75.0** |
| 5× | 0.83 | 114.5 | 64.0 | 59.0 | **59.6** |

- **Answer:** quantile models are the cheapest at every cost ratio.
- Quantile LightGBM saves **2% to 40%** compared with the normal forecast. The more a missed order costs, the
  bigger the saving. *Example:* at ratio 5, the cost drops from 100 to about 60, a 40% saving.
- It also beats the safety stock rival by **3% to 7%** at every ratio, and the corrected DM test says this is real
  (p below 0.05 for all ratios).
- The ranges are narrow. *Example:* at ratio 3, the saving is 25.0%, with a range of 24.1% to 25.8%.
- The cheapest level in the data is close to the newsvendor rule. *Example:* at ratio 1 the cheapest level is 0.5,
  and at ratio 4 it is 0.8, exactly as the rule says.

### Compared with Panda & Mohanty (same data, same test weeks 136–145)

| | RMSE | MAE |
|---|---|---|
| Their best tree model (CatBoost) | 186.7 | 99.4 |
| My LightGBM | 105.7 | 62.4 |

My tree models make about **40% smaller** errors (RMSE and MAE). Their RMSLE is lower (0.29 vs my 0.48), but their
steps are not fully explained, so that number is hard to compare directly.

---

## 9. Limitations (what my study cannot show)

| Limitation | Simple meaning | Example |
|---|---|---|
| **Costs are estimated, not real** | The data has no prices for waste or missed orders. | I test ratios from 0.5 to 5 instead of using one real price. |
| **Only one dataset** | Results may be different for other companies. | A bakery or a supermarket may behave differently. |
| **Simple rules** | Each week is treated on its own. | In real life, some food can be frozen and used next week. |
| **Recorded orders = true demand** | If food ran out, some demand was never recorded. | If 120 people wanted a meal but only 100 were made, the data may show 100. |
| **Missing weeks are skipped** | I assume a missing week means "not on sale". | If it really meant "zero orders", I would miss some zeros. |
| **Only 30 test weeks** | A small number of test weeks. | That is why I used the corrected test and the bootstrap. |
| **No deep learning** | LSTM and similar models were not tried. | Because of the 12-week time limit. |
| **Old, anonymous data** | No real dates or city names. | So I cannot add holidays or weather. |

---

## 10. Questions people may ask, with short answers

### About the problem and the gap

**Q: Why is this problem important?**
A: About 1.05 billion tonnes of food were wasted in 2022, and restaurants and canteens caused about 28% of it.
Better orders mean less waste and fewer unhappy customers.

**Q: What is new in your work?**
A: Three things. First, quantile models on this public food data. Second, turning forecasts into waste, shortage and
cost. Third, fair testing over time that anyone can repeat with the public data and code.

**Q: How is your work different from Panda & Mohanty?**
A: They only checked how close the forecasts were. I also check if forecasts are too high or too low, use quantile
models, estimate the cost, and test in three rounds instead of one.

### About the data

**Q: Why this dataset?**
A: It is a real weekly planning problem, it is public, it has no personal data, and another study used it, so I can compare.

**Q: Why did you not fill missing weeks with zero?**
A: The smallest recorded week is 13 orders, and there are no zeros at all. So a missing week most likely means the
meal was not on sale.

**Q: Why did you keep the very large weeks?**
A: They are real orders from big centres and special offers. *Example:* a meal on a home-page offer can sell 3 times
more. If I removed these weeks, the model would be worst exactly when it matters most.

**Q: Why the log scale?**
A: Orders go from 13 to over 24,000. The log scale makes big numbers smaller (100 → 4.6, 20,000 → 9.9), so a few
huge weeks do not take over the model. At the end, I change the answers back into meals.

**Q: Is it fair to use the price and offer of the week you forecast?**
A: Yes. The company plans prices and offers in advance. *Example:* the kitchen already knows next week's email offer
when it orders the food.

### About the method

**Q: How do you make sure there is no data leakage?**
A: Every input uses only weeks before the forecast week. Settings are chosen on learning weeks only. The best level
comes from the newsvendor rule, not from the test weeks. And an automatic test checks the inputs.

**Q: Why not a random train/test split?**
A: A random split lets the model learn from future weeks. *Example:* it could learn from week 140 and then be
tested on week 130. That is impossible in real life. My way copies how a real planner works.

**Q: Why 1 to 10 weeks ahead?**
A: Weeks 1–2 help to buy ingredients. Weeks 3–10 help to plan with suppliers. Panda & Mohanty also used 10 weeks.

**Q: Why one model for everything, not one per meal?**
A: 3,597 small models would each have very little data. One big model learns from all of them together.
*Example:* a new meal with only 10 weeks of history can learn from similar meals that have 145 weeks.

**Q: How did you choose the model settings?**
A: In each round, I tried a few settings on the last 10 learning weeks and kept the best one. XGBoost and LightGBM
also stop adding trees when they stop improving.

### About the models

**Q: Why tree models?**
A: They are the strongest on table data, they handle many kinds of inputs, they are fast, and LightGBM won the M5 contest.

**Q: Why not LSTM or other deep learning?**
A: Research shows tree models still beat deep learning on table data. Deep learning is slower and harder to set up,
and I had 12 weeks. It is future work.

**Q: What is a quantile model, in one sentence?**
A: A normal model says "about 100 meals"; a quantile model says "130 meals will be enough in 8 out of 10 weeks",
and you choose how safe you want to be.

**Q: Why not just add a safety stock to the normal forecast?**
A: I tested that too. Quantile LightGBM was still 3% to 7% cheaper at every cost ratio, and the corrected test
says the difference is real.

**Q: What is "crossing", and how did you fix it?**
A: Sometimes the 0.7 answer is bigger than the 0.8 answer, which makes no sense. *Example:* 0.7 says 130 and 0.8
says 125. I sort the answers so they always go up. Quantile XGBoost gives all levels from one model, so it had no crossing.

### About the results

**Q: Which model is best?**
A: For accuracy, LightGBM and XGBoost (a tie, about 28.6% error). For cost, quantile LightGBM or quantile XGBoost at
the level the newsvendor rule gives.

**Q: Is the difference real, or just luck?**
A: I used the Diebold–Mariano test, corrected for linked weeks and the small number of test weeks, plus the
bootstrap. The main differences are real (p < 0.05), and the ranges are narrow.

**Q: Why are all models too low on average?**
A: Most weeks are small, but a few are huge (offers, big centres). A model that aims for the typical week misses the
huge ones. *Example:* if a meal usually sells 100 but sells 400 in an offer week, a "middle" forecast is far too low
in that week. This is exactly why quantile models help.

**Q: What does 28.6% error mean in practice?**
A: If all the real orders add up to 1,000 meals, the forecasts are wrong by about 286 meals in total. Linear
Regression is wrong by about 320, so LightGBM is about 10% better.

**Q: Your RMSLE is higher than Panda & Mohanty's. Why?**
A: My RMSE and MAE are about 40% lower on the same test weeks. Their steps are not fully explained, and some of their
numbers do not match each other, so RMSLE is hard to compare directly.

**Q: Is the saving real money?**
A: No. It is an estimate under clear rules and a range of costs. The main finding is the pattern: the more a missed
order costs, the more a quantile model saves.

**Q: What should a company do with this?**
A: First, decide how much a missed order costs compared with a wasted meal. Then order at the matching level with a
quantile model. *Example:* if a missed order costs 3 times more, order at level 0.75. The app shows this.

### About limitations and future work

**Q: What is the biggest weakness?**
A: The costs are estimated, not real, and only one dataset is used.

**Q: Would the results work for another company?**
A: The method can be used anywhere, but the exact numbers may change. Testing on more datasets is future work.

**Q: What would you do next?**
A: Use real cost data from a company; test more datasets; try deep learning; allow food to be kept for more than
one week; and handle hidden demand when food runs out.

**Q: Can I repeat your work?**
A: Yes. The data is public, all the code is in this repository, each step has its own notebook, and the README
explains how to run everything.

---

*All numbers come from the tables in `results/tables/`. Full references are in the project report.*
