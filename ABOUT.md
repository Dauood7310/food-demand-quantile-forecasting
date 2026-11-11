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
| **Over-forecast %** | How often the forecast was higher than the real orders. 50% = balanced. | 46% means the guess was too high in 46 of 100 cases and too low in 54. |
| **Middle value (median) vs average** | The middle value is the "normal" week; the average is pulled up by a few huge weeks. | Weeks of 100, 100, 100 and 500: the middle value is 100, but the average is 200. |
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

This part explains every result, step by step. You do not need to know machine learning to read it.

### 8.1 Before you read the numbers

**What was tested?** The models made **98,475 forecasts** in the 3 test rounds. Each forecast is "how many of
this meal will this centre sell in this week?". In those test weeks, people really ordered about **24.5 million meals**.
A normal forecast is for about **135 meals** (the middle value), but some are for thousands.

**How do we score a forecast?** We compare the guess with what really happened.

> **Example:** the model guessed 90 meals, and 100 were really ordered. The **error** is 10 meals.

Because there are 98,475 forecasts, we need a way to put all the errors into one number. There are several
ways, and each one tells us something a little different:

| Score | How it is worked out | Example | Good value |
|---|---|---|---|
| **WAPE** (main score) | Add up all the errors, divide by all the real orders | Real orders: 100 + 200 = 300. Errors: 10 + 50 = 60. WAPE = 60 ÷ 300 = **20%** | Low. 0% = perfect |
| **MAE** | The average error, in meals | Errors of 10 and 50 → MAE = **30 meals** | Low |
| **RMSE** | Like MAE, but big errors are punished more | Errors of 10 and 50 → RMSE = **36 meals** (the 50 counts extra) | Low |
| **RMSLE** | The error in "percent-like" terms, so small and big meals count the same | Guessing 20 when it was 10 is as bad as guessing 2,000 when it was 1,000 | Low |
| **Bias** | The average of (guess − real). Shows the direction of the error | Bias −20 = on average the guess is 20 meals **too low** | Near 0 |
| **Over-forecast %** | How often the guess was above the real number | 46% = too high 46 times in 100, too low 54 times | Near 50% |

**Why is WAPE the main score?** It is easy to read ("wrong by X meals in every 100") and it is fair to both small
and big meals. RMSE and RMSLE are included because the earlier study (Panda & Mohanty) used them.

**"Average of 3 rounds".** Every model was tested three times, on three different sets of weeks (see section 7).
The tables show the average of the three tests, so one lucky or unlucky set of weeks cannot decide the result.

---

### 8.2 Step 1 – The three simple models (notebook 04)

These are the "easy rivals". A smart model must do better than them, or it is not worth using.

| Method | How it guesses | WAPE | RMSE | MAE | RMSLE | Bias | Over-forecast % |
|---|---|---|---|---|---|---|---|
| Copy last week | Last week 150, so guess 150 | 50.9% | 294 | 127 | 0.81 | −11 | 50% |
| 26-week average | Average of the last 26 weeks | 38.4% | 237 | 96 | 0.62 | −23 | 52% |
| **Linear Regression** | Past orders + a bit for offers − a bit for a high price | **32.0%** | **183** | **80** | **0.51** | −28 | 46% |

**How to read it:**

- **Copy last week: 50.9%.** For every 100 meals people ordered, this method is wrong by about 51 meals. That is
  very poor: it gets about half the meals wrong.
- **26-week average: 38.4%.** Wrong by about 38 meals in every 100. Better, because an average is calmer than one single week.
- **Linear Regression: 32.0%.** Wrong by about 32 meals in every 100. This is the best simple method.

**What we learned:**

- **Linear Regression is the best simple model in every round** (31.0% to 33.0%). So 32% is the score the smart models must beat.
- **Copying last week only works for next week.** It is wrong by 34 in 100 for next week, but by 56 in 100 for ten
  weeks ahead. *Why:* orders jump up and down a lot, so last week tells you little about ten weeks later.
- **Linear Regression stays steady** (about 30 to 34 in 100) for every week ahead, because it knows the planned price and offers.
- **All three guess too low** (the bias is negative). *Why:* the models aim for the "normal" week, not the average.
  *Example:* a meal sells 100, 100, 100 and then 500 in an offer week. The normal week is 100, but the average is 200.
  A guess near 100 is right for normal weeks but far too low overall.

---

### 8.3 Step 2 – The tree models (notebook 05) – answers RQ1

**RQ1: Are tree models more accurate than simple methods?**

| Model | Type | WAPE | RMSE | MAE | RMSLE | Bias | Over-forecast % |
|---|---|---|---|---|---|---|---|
| Copy last week | Simple | 50.9% | 294 | 127 | 0.81 | −11 | 50% |
| 26-week average | Simple | 38.4% | 237 | 96 | 0.62 | −23 | 52% |
| Linear Regression | Simple | 32.0% | 183 | 80 | 0.51 | −28 | 46% |
| Random Forest | Tree | 29.4% | 164 | 73 | 0.49 | −26 | 45% |
| XGBoost | Tree | 28.7% | 152 | 71 | 0.49 | −23 | 46% |
| **LightGBM** | Tree | **28.6%** | **152** | **71** | **0.49** | −22 | 46% |

**Answer: yes.** All three tree models beat all three simple methods.

**How to read it:**

- **LightGBM: 28.6%.** For every 100 meals ordered, LightGBM is wrong by about 29. Linear Regression is wrong by 32,
  and copying last week by 51.
- **Is 29 in 100 good?** Yes, for this problem. The model guesses up to 10 weeks ahead, for one meal at one centre,
  and orders jump a lot because of offers. A guess for one meal in one place is always much less exact than a guess
  for a whole company's total.
- **MAE 71.** A typical LightGBM guess is off by about 71 meals. *Example:* real orders 300, guess 229.
- **RMSE 152** is bigger than MAE 71 because a few very large misses (offer weeks) count extra in RMSE.
- **How much better?** LightGBM's error is **10.5% smaller** than Linear Regression's (32.0 → 28.6) and
  **44% smaller** than copying last week's (50.9 → 28.6).

**The same result in each of the 3 rounds (WAPE):**

| Model | Round 1 (weeks 116–125) | Round 2 (126–135) | Round 3 (136–145) | Average |
|---|---|---|---|---|
| Copy last week | 52.1% | 52.2% | 48.5% | 50.9% |
| 26-week average | 39.1% | 41.9% | 34.2% | 38.4% |
| Linear Regression | 32.0% | 33.0% | 31.0% | 32.0% |
| Random Forest | 29.0% | 31.3% | 27.9% | 29.4% |
| XGBoost | 27.7% | 30.7% | 27.5% | 28.7% |
| LightGBM | 27.7% | 30.8% | 27.4% | 28.6% |

The order of the models is the same in every round. So the result is stable, not luck from one set of weeks.
Round 2 is the hardest for every model, and round 3 the easiest.

**Does the error grow when we look further ahead? (WAPE by weeks ahead)**

| Weeks ahead | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Copy last week | 33.8 | 44.7 | 54.7 | 53.0 | 52.4 | 49.5 | 54.8 | 56.8 | 54.4 | 55.8 |
| Linear Regression | 31.8 | 29.7 | 30.9 | 29.9 | 31.4 | 31.8 | 33.3 | 34.4 | 33.6 | 33.4 |
| LightGBM | 26.5 | 27.2 | 28.1 | 27.7 | 28.2 | 28.0 | 29.7 | 29.7 | 30.4 | 31.7 |

*How to read it:* guessing next week, LightGBM is wrong by about 27 in 100. Guessing ten weeks ahead, by about 32
in 100. The further ahead, the harder, but LightGBM is the best at every distance.

**Is the difference real, or just luck?** A statistical test (the Diebold–Mariano test) answers this. It gives a
"p-value". A small p-value means the difference is very unlikely to be just luck. Below 0.05 is accepted as real.

| Comparison | p-value | Meaning |
|---|---|---|
| LightGBM vs Linear Regression | below 0.001 | Really better (very unlikely to be luck) |
| XGBoost vs Linear Regression | below 0.001 | Really better |
| Random Forest vs Linear Regression | below 0.001 | Really better |
| LightGBM vs Random Forest | 0.002 | Really better |
| LightGBM vs XGBoost | 0.96 | **No real difference: a tie** |

**What does LightGBM look at most?** (share of its "decisions")

| Input | Share | Simple meaning |
|---|---|---|
| Average orders of the last 26 weeks | 43.6% | "How much does this meal usually sell here?" |
| Average orders of all past weeks | 36.9% | The long-term level |
| Price compared with the usual price | 4.7% | "Is it cheaper than normal this week?" |
| Email offer in that week | 3.7% | "Is it being advertised?" |
| Which meal it is | 2.9% | Some meals behave differently |
| Home-page offer in that week | 1.7% | "Is it shown on the home page?" |
| Last week's orders | 0.6% | Almost not used |

*How to read it:* the model mostly asks "what does this meal normally sell?", then adjusts for price and offers.
Last week alone matters very little, which matches the poor result of "copy last week".

**One problem remains: the guesses are too low.** LightGBM is above the real orders only 46 times in 100, and on
average 22 meals too low. If a kitchen made exactly what LightGBM says, **food would run out more often than it
would be wasted**. The next step fixes this.

---

### 8.4 Step 3 – The "safe-level" (quantile) models (notebook 06) – answers RQ2

**RQ2: Do quantile models change how often we guess too high or too low? And can we trust their levels?**

**The idea again.** A normal model gives one guess ("100 meals"). A quantile model gives a guess for a **level** you choose:

> **Example for one meal and week:**
> - Level 0.3 → "80 meals" (enough in about 3 of 10 weeks, a small order)
> - Level 0.5 → "100 meals" (enough in about 5 of 10 weeks, the middle)
> - Level 0.8 → "130 meals" (enough in about 8 of 10 weeks, a safe order)
> - Level 0.9 → "160 meals" (enough in about 9 of 10 weeks, a very safe order)

**Test 1 – Coverage: do the levels keep their promise?**
Coverage = in how many cases the real orders were at or below the forecast (the forecast was "enough").

| Level you ask for | Linear QR | Quantile XGBoost | Quantile LightGBM | Normal LightGBM |
|---|---|---|---|---|
| 0.3 (should be 30%) | 29.8% | 30.4% | 30.6% | 45.9% |
| 0.4 (should be 40%) | 40.0% | 40.0% | 40.2% | 45.9% |
| 0.5 (should be 50%) | 50.0% | 49.7% | 49.3% | 45.9% |
| 0.6 (should be 60%) | 60.1% | 59.0% | 58.0% | 45.9% |
| 0.7 (should be 70%) | 69.8% | 68.6% | 67.5% | 45.9% |
| 0.8 (should be 80%) | 79.8% | 77.9% | 77.4% | 45.9% |
| 0.9 (should be 90%) | 89.7% | 88.2% | 87.4% | 45.9% |

**Answer: yes.**

- *How to read it:* you ask quantile LightGBM for level 0.8, hoping the number is enough in 80% of cases. It was
  enough in 77.4%. That is close to the promise. At the high levels the tree models are 2–3 points under the target.
- **The normal LightGBM is always at 45.9%.** It gives one number only, so you cannot ask it to be safer.
- So quantile models **do change the balance**: you decide how often you want to be "too high" (safe) or "too low".

**Test 2 – Pinball loss: how good is each level?**
Pinball loss is the score for a level forecast. It punishes the mistake that matters more for that level.
*Example:* for level 0.8 you want to be safe, so being too low is punished 4 times more than being too high.
Lower is better.

| Level | Linear QR | Quantile XGBoost | Quantile LightGBM | Normal LightGBM |
|---|---|---|---|---|
| 0.3 | 31.4 | 29.3 | **29.2** | 31.4 |
| 0.4 | 36.2 | 33.4 | **33.2** | 33.5 |
| 0.5 | 38.9 | 35.3 | **35.0** | 35.7 |
| 0.6 | 39.2 | 35.2 | **35.0** | 37.8 |
| 0.7 | 37.1 | **32.8** | 33.0 | 40.0 |
| 0.8 | 31.8 | **27.8** | 28.1 | 42.1 |
| 0.9 | 22.0 | **19.2** | 19.5 | 44.3 |
| **Average** | 33.8 | **30.4** | **30.4** | 37.8 |

- *How to read it:* the two tree quantile models have the lowest score at every level. They are tied on average (30.4).
- The normal forecast gets **much worse at high levels** (44.3 at level 0.9, against 19.5). One fixed guess cannot
  act as a safe order.
- The test says the difference between quantile LightGBM and the normal forecast is real (p below 0.001 at levels 0.5, 0.7 and 0.9).

**A small extra finding.** The level-0.5 forecast of quantile LightGBM is also a slightly better "normal" guess
than the normal LightGBM: WAPE 28.1% against 28.6%, and bias −15 against −22.

**"Crossing".** Quantile LightGBM builds a separate model for each level. In about 1 of 4 forecasts, two levels
came out in the wrong order (*example:* level 0.7 says 130 but level 0.8 says 125). The notebook sorts the numbers
so they always go up. Quantile XGBoost builds all levels in one model and had no crossing.

---

### 8.5 Step 4 – Waste, shortage and cost (notebook 07) – answers RQ3

**RQ3: When costs change, which model and level give the lowest cost?**

**How the cost is worked out** (the data has no real prices, so we use simple rules):

> **Example:** the kitchen makes what the forecast says: 100 meals.
> - Only 80 are ordered → **20 wasted**. Each wasted meal costs **1**. Cost = 20.
> - 120 are ordered → **20 missed orders**. Each missed order costs **1 × the cost ratio**.
>   If the ratio is 3, cost = 20 × 3 = 60.

**The cost ratio** says how many times worse a missed order is than a wasted meal. We do not know the real
ratio, so we test seven: 0.5, 1, 1.5, 2, 3, 4 and 5.

**Which level should we order at?** An old stock rule (the "newsvendor rule") gives the answer:

| A missed order costs… | 0.5× a wasted meal | 1× | 1.5× | 2× | 3× | 4× | 5× |
|---|---|---|---|---|---|---|---|
| Best level to order at | 0.33 | 0.50 | 0.60 | 0.67 | 0.75 | 0.80 | 0.83 |

*Example:* if running out is 4 times worse than waste, order the number that is enough in 80% of weeks (level 0.80).

**Result 1 – Total cost of each way of ordering** (normal LightGBM = 100; below 100 is cheaper):

| A missed order costs… | Normal Linear Regression | Normal LightGBM | Linear QR | Quantile XGBoost | **Quantile LightGBM** |
|---|---|---|---|---|---|
| 0.5× | 109.7 | 100 | 103.6 | 96.0 | **95.7** |
| 1× | 111.6 | 100 | 108.9 | 99.0 | **98.1** |
| 1.5× | 112.6 | 100 | 103.6 | 93.2 | **92.5** |
| 2× | 113.2 | 100 | 97.0 | **86.2** | **86.2** |
| 3× | 113.8 | 100 | 84.9 | **74.5** | 75.0 |
| 4× | 114.2 | 100 | 75.4 | **66.0** | 66.7 |
| 5× | 114.5 | 100 | 67.8 | **59.0** | 59.6 |

**Answer: the tree quantile models are the cheapest at every cost ratio.**

- *How to read it:* take the row "3×". If a kitchen spends £100 using the normal LightGBM forecast, it would spend
  only about **£75** using quantile LightGBM. That is a saving of 25%.
- **The more a missed order costs, the bigger the saving:** 2% at ratio 1, 14% at ratio 2, 25% at ratio 3, 40% at ratio 5.
- **Normal Linear Regression is always the most expensive** (10–15% more than normal LightGBM). So a more accurate
  model does save money, but ordering at the right level saves much more.
- Quantile XGBoost and quantile LightGBM are almost the same.

**Result 2 – What happens to waste and shortage?**
(Waste % and shortage % are shares of all meals ordered. "No shortage" = how often nobody went without a meal.)

| A missed order costs… | Way of ordering | Waste % | Shortage % | No shortage in… |
|---|---|---|---|---|
| any | Normal LightGBM | 10.1% | 18.6% | 46 of 100 cases |
| 0.5× | Quantile LightGBM (level 0.33) | 6.1% | 24.8% | 35 of 100 |
| 1× | Quantile LightGBM (level 0.50) | 11.2% | 16.9% | 50 of 100 |
| 2× | Quantile LightGBM (level 0.67) | 18.5% | 11.1% | 65 of 100 |
| 3× | Quantile LightGBM (level 0.75) | 24.0% | 8.5% | 73 of 100 |
| 5× | Quantile LightGBM (level 0.83) | 32.0% | 5.9% | 82 of 100 |

- *How to read it:* the normal forecast always wastes 10 meals and misses 19 orders in every 100, whatever the costs are.
- The quantile forecast **moves with the costs**. When missed orders are expensive (ratio 5), it makes more food:
  waste rises to 32 in 100, but missed orders fall to 6 in 100, and customers are fully served in 82 of 100 cases.
- When waste is the bigger problem (ratio 0.5), it makes less food: waste falls to 6 in 100.
- **This is the trade-off:** you cannot cut waste and shortage at the same time. You choose which one hurts more,
  and the quantile model gives the right order for that choice.

**Result 3 – Does the stock rule really work on this data?**
We tried ordering at every level and looked for the cheapest one.

| A missed order costs… | Level the rule says | Cheapest level found |
|---|---|---|
| 0.5× | 0.33 | 0.4 |
| 1× | 0.50 | 0.5 |
| 1.5× | 0.60 | 0.6 |
| 2× | 0.67 | 0.7 |
| 3× | 0.75 | 0.8 |
| 4× | 0.80 | 0.8 |
| 5× | 0.83 | 0.9 |

The cheapest level is the same as the rule, or the nearest level we tested. So the rule works on real meal data.

---

### 8.6 Step 5 – Extra checks: can we trust the savings? (notebook 08)

**Check 1 – A fairer rival: the safety stock.**
Real planners do not order exactly the forecast. They add "a little extra just in case" (a safety stock).
So we built this rival too: the normal LightGBM forecast plus an extra amount learned from its own past mistakes.
*Example:* if the model was usually about 15% too low in the past weeks, add about 15%.

| A missed order costs… | Normal LightGBM | Safety stock | Quantile LightGBM | Saving vs safety stock |
|---|---|---|---|---|
| 0.5× | 100 | 99.8 | **95.7** | 4.1% |
| 1× | 100 | 101.5 | **98.1** | 3.3% |
| 1.5× | 100 | 95.7 | **92.5** | 3.4% |
| 2× | 100 | 89.3 | **86.2** | 3.4% |
| 3× | 100 | 78.5 | **75.0** | 4.5% |
| 4× | 100 | 70.3 | **66.7** | 5.1% |
| 5× | 100 | 64.0 | **59.6** | 6.8% |

- *How to read it:* the safety stock is a good trick. It gets most of the saving (for ratio 5: from 100 down to 64).
- But **quantile LightGBM is still 3% to 7% cheaper** at every ratio. *Why:* the safety stock adds the same share to
  every meal, while the quantile model knows which meals are steady and which are jumpy, and adds more only where needed.

**Check 2 – How sure are we? (the 95% range)**
We picked the 3,597 meal–centre pairs again at random, 1,000 times, and worked out the saving each time
(this is called a bootstrap). The range shows where the saving falls 95% of the time.

| A missed order costs… | Saving vs normal forecast (range) | Saving vs safety stock (range) |
|---|---|---|
| 0.5× | 4.3% (3.6 to 5.0) | 4.1% (3.6 to 4.7) |
| 1× | 1.9% (1.4 to 2.4) | 3.3% (2.8 to 3.8) |
| 1.5× | 7.5% (6.8 to 8.1) | 3.4% (2.8 to 3.9) |
| 2× | 13.8% (13.0 to 14.5) | 3.4% (2.8 to 4.2) |
| 3× | 25.0% (24.1 to 25.8) | 4.5% (3.6 to 5.4) |
| 4× | 33.3% (32.4 to 34.1) | 5.1% (4.0 to 6.2) |
| 5× | 40.4% (39.5 to 41.2) | 6.8% (5.6 to 8.1) |

- *How to read it:* at ratio 3, the saving is 25.0%, and we are fairly sure the true saving is between 24.1% and 25.8%.
- **Every range is above zero.** So in every case, quantile LightGBM really is cheaper.

**Check 3 – A stricter test.**
The first test for "real or luck?" assumed each week is separate. But a busy week is often followed by another
busy week, and we only have 30 test weeks. So we repeated every test with a stricter version (the "corrected"
Diebold–Mariano test). **All the main results stayed real:**

- tree models vs Linear Regression: p below 0.001;
- quantile LightGBM vs normal forecast, on cost: p below 0.001 at every ratio;
- quantile LightGBM vs safety stock, on cost: p is 0.03 or lower at every ratio (all below 0.05).

---

### 8.7 Compared with the earlier study (Panda & Mohanty)

They used the same data and the same test weeks (136–145), so this is a fair comparison.

| | RMSE | MAE |
|---|---|---|
| Their Random Forest | 228.9 | 115.9 |
| Their LightGBM | 194.7 | 104.5 |
| Their XGBoost | 205.3 | 106.2 |
| Their best tree model (CatBoost) | 186.7 | 99.4 |
| **My LightGBM (round 3)** | **105.7** | **62.4** |

- *How to read it:* on the same weeks, my LightGBM is off by about 62 meals on average, and their best tree model by about 99.
  My errors are about **40% smaller**.
- Their RMSLE is lower (0.29 against my 0.48). But their paper does not fully explain how the inputs were built,
  and some of its numbers do not match each other, so that one score is hard to compare.
- Their deep learning models (LSTM) seem to be reported on a different scale, so they cannot be compared with these numbers.

---

### 8.8 All the results in six lines

1. **Simple methods:** the best is Linear Regression, wrong by 32 meals in every 100.
2. **Tree models are better:** LightGBM and XGBoost are wrong by about 29 in every 100, and the difference is real.
3. **But all normal forecasts are a bit too low,** so food would run out more often than it is wasted.
4. **Quantile models fix this:** you choose a safety level, and the level keeps its promise (0.8 → enough in about 77–80% of cases).
5. **Ordering at the right level saves money:** 2% to 40% against the normal forecast, more when missed orders are costly.
6. **The saving is real:** it is still 3% to 7% against a safety stock, the ranges are narrow, and the stricter test agrees.

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
