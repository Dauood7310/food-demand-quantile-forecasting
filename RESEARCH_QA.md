# Research questions and answers

**Machine Learning for Demand Forecasting in Perishable Food Production Planning**
Mohammad Dauood · COM748 MSc Research Project · Supervisor: Dr Serif Salem

This file lists the questions a supervisor, examiner or conference audience may ask, with short answers in
very easy English. For the full explanation of the words and results, see [ABOUT.md](ABOUT.md).

**Contents**

1. [The project in short](#1-the-project-in-short)
2. [The problem and why it matters](#2-the-problem-and-why-it-matters)
3. [Other studies and the research gap](#3-other-studies-and-the-research-gap)
4. [The data](#4-the-data)
5. [Preparing the data and building inputs](#5-preparing-the-data-and-building-inputs)
6. [The method and the testing plan](#6-the-method-and-the-testing-plan)
7. [The models](#7-the-models)
8. [The scores (metrics)](#8-the-scores-metrics)
9. [Results: accuracy (RQ1)](#9-results-accuracy-rq1)
10. [Results: quantile models (RQ2)](#10-results-quantile-models-rq2)
11. [Results: waste, shortage and cost (RQ3)](#11-results-waste-shortage-and-cost-rq3)
12. [Statistics: is it real or luck?](#12-statistics-is-it-real-or-luck)
13. [Hard and critical questions](#13-hard-and-critical-questions)
14. [Limitations](#14-limitations)
15. [Using the results in real life](#15-using-the-results-in-real-life)
16. [Ethics and repeating the work](#16-ethics-and-repeating-the-work)
17. [Future work](#17-future-work)
18. [Personal questions](#18-personal-questions)

---

## 1. The project in short

**Q: Can you explain your project in one minute?**
A: A meal company must buy fresh food before it knows how many people will order. If it buys too much, food is
wasted. If it buys too little, customers get nothing. I forecast weekly meal orders with machine learning, and
then I check how much waste, shortage and cost each forecast would cause. The main finding: the most accurate
model is not always the cheapest. A "quantile" model, which gives a safer number, saves up to 40% of the cost
when missed orders are expensive.

**Q: What is your main contribution?**
A: Three things. (1) The first use of quantile models on the public Genpact meal data. (2) A link from forecasts to
waste, shortage and cost, with public data that anyone can check. (3) Fair testing over time, with no data leakage,
plus a strong rival (safety stock) and proper statistical tests.

**Q: What is your one key message?**
A: Do not only ask "how close is the forecast?". Ask "what order does it lead to, and what does that order cost?".

**Q: Who can use this work?**
A: Planners in meal delivery, canteens, bakeries and any business with fresh products that cannot be kept.

**Q: What are your research questions?**
A: RQ1: Are tree models more accurate than simple methods? RQ2: Do quantile models change how often we forecast too
high or too low, and can we trust their levels? RQ3: When costs change, which model and level give the lowest cost?

**Q: What are the short answers to them?**
A: RQ1: Yes, LightGBM is wrong by about 29 meals in every 100, against 32 for Linear Regression. RQ2: Yes, a level of
0.8 is enough in about 77–80% of cases, as promised. RQ3: Quantile LightGBM or XGBoost, at the level given by the
newsvendor rule, is the cheapest at every cost ratio.

---

## 2. The problem and why it matters

**Q: Why is this problem important?**
A: About 1.05 billion tonnes of food were wasted in 2022 (UNEP), and restaurants and canteens caused about 28% of it.
Better orders mean less waste, less lost money and fewer unhappy customers.

**Q: What is wrong with normal forecasting studies?**
A: They only measure how close the forecast is. For them, 20 meals too many and 20 too few are the same mistake.
*Example:* a wasted meal may cost £3 of food, but a missed order may lose a £12 sale. These are not the same.

**Q: What does "perishable" mean, and why does it matter?**
A: It means the food goes bad quickly. *Example:* a salad made on Monday cannot be sold next Monday. So every week
starts from zero, and a wrong order cannot be fixed later.

**Q: Why forecast weekly, and not daily?**
A: The data is weekly, and ingredient orders in this business are also planned weekly.

**Q: Why forecast 1 to 10 weeks ahead?**
A: Weeks 1–2 help to buy ingredients. Weeks 3–10 help to plan with suppliers. The earlier study also used 10 weeks,
so I can compare.

---

## 3. Other studies and the research gap

**Q: What did other studies find?**
A: (1) Better forecasts can cut catering waste by 14% to 52%. (2) Tree models like LightGBM are the strongest on
table data and won the M5 contest. (3) The "best" forecast depends on how you score it. (4) Many papers have data
leakage, so fair testing is very important.

**Q: What is the research gap?**
A: Three gaps. (1) No quantile models on this public food data. (2) Studies that measure food waste use private
data, so nobody can repeat them. (3) No fair, public study links forecasts to waste, shortage and cost.

**Q: How is your work different from Panda & Mohanty (2023)?**
A: They used the same data but only checked accuracy, with one train/test split. I use three test rounds, quantile
models, a cost estimate, a safety-stock rival and statistical tests.

**Q: What is the newsvendor model?**
A: An old stock rule. *Example:* a newspaper seller must decide how many papers to buy each morning. If a missed
sale costs more than an unsold paper, he should buy extra. The rule tells exactly how much extra.

**Q: Is the newsvendor rule your idea?**
A: No, it is a classic rule from operations research. My work tests it on real meal data with machine learning forecasts.

**Q: What is the M5 competition, and why do you mention it?**
A: A big public forecasting contest on Walmart sales (2020). LightGBM won it, and its second part used quantiles.
It shows that my models are the standard, proven choice.

---

## 4. The data

**Q: Which data did you use?**
A: The public "Food Demand Forecasting" data from Genpact, on Kaggle. It has 456,548 rows: weekly orders for
51 meals at 77 centres over 145 weeks, with price, offers, centre details and meal details.

**Q: Why this dataset?**
A: It is a real weekly planning problem, it is public, it has no personal data, and an earlier study used it.

**Q: What does one row look like?**
A: One meal at one centre in one week. *Example:* in week 1, centre 55 sold 177 of meal 1885 at a price of 136.83,
with no offer.

**Q: Is the data clean?**
A: Yes. No empty values, no repeated rows, and no weeks with zero orders.

**Q: Are there missing weeks?**
A: Yes, 65,017 meal–centre–weeks have no row. I did not fill them with zero.

**Q: Why did you not fill missing weeks with zero?**
A: The smallest recorded value is 13 orders and there are no zeros at all. So a missing week most likely means the
meal was not on sale, not that nobody wanted it.

**Q: Are there outliers (very large values)?**
A: Yes, about 7.2% of rows. Orders go from 13 to 24,299, with a middle value of 136.

**Q: Why did you keep the outliers?**
A: They are real orders from big centres and special offers. *Example:* a meal on the home page can sell about
3 times more. Removing them would make the model blind in exactly the weeks that matter most.

**Q: Is there a yearly pattern (seasonality)?**
A: No. I checked the totals week by week and found no pattern that repeats every 52 weeks.

**Q: Why did you not use the competition's test file?**
A: It has no real order numbers, so forecasts for it cannot be checked.

**Q: Do you know the real dates or cities?**
A: No. Weeks are numbered 1 to 145, and cities are codes. So I cannot add holidays or weather.

---

## 5. Preparing the data and building inputs

**Q: What preparation did you do?**
A: I joined the three files, created the inputs from past weeks, changed orders to the log scale, and turned text
columns into categories.

**Q: What is the log scale, and why use it?**
A: A maths tool that makes big numbers much smaller but keeps small numbers almost the same. *Example:* 10 → 2.4,
100 → 4.6, 1,000 → 6.9, 20,000 → 9.9. Without it, a few huge weeks would take over the model. At the end, the
answers are changed back into meals.

**Q: How many inputs (features) do you have, and what are they?**
A: 25. The number of weeks ahead; past orders (last week, averages of 4, 8 and 26 weeks and of all weeks, how much
orders jump, how long the history is); the price, discount and offers of the target week; and the centre and meal details.

**Q: Which inputs matter most?**
A: The long-term averages of past orders (about 80% of LightGBM's decisions), then the price compared with the usual
price (4.7%) and email offers (3.7%). Last week's orders matter very little (0.6%).

**Q: What is data leakage, and how did you avoid it?**
A: Using future information by mistake, like seeing exam answers before the exam. Every input uses only weeks up to
the day of the forecast. Model settings are chosen on training weeks only. An automatic test checks this.

**Q: Is it fair to use the price and offer of the week you forecast?**
A: Yes. The company plans prices and offers in advance. *Example:* the kitchen already knows next week's email offer
when it orders the food.

**Q: Why one model for all meals and centres ("global model")?**
A: 3,597 small models would each have very little data. One model learns from all of them (4.4 million examples).
*Example:* a new meal with 10 weeks of history can learn from similar meals with 145 weeks.

**Q: How does one model forecast 1 to 10 weeks ahead?**
A: "Weeks ahead" is one of the inputs. The same model is asked: "what about 1 week ahead? 2 weeks? … 10 weeks?".

---

## 6. The method and the testing plan

**Q: What is your methodology?**
A: CRISP-DM, a common step-by-step plan for data projects: understand the problem, understand the data, prepare it,
build models, test them, and use them. I added one step: the cost estimate.

**Q: How did you test the models?**
A: With three rounds over time ("rolling-origin"). Round 1: learn from weeks 1–115, test on 116–125. Round 2: learn
from 1–125, test on 126–135. Round 3: learn from 1–135, test on 136–145.

**Q: Why not a random train/test split?**
A: A random split lets the model learn from the future. *Example:* it could learn from week 140 and be tested on
week 130. That cannot happen in real life, and it makes results look better than they are.

**Q: Why three rounds, not one?**
A: One round could be lucky or unlucky. With three, I can see that the ranking of the models is the same each time.

**Q: Why only three rounds, not more?**
A: Each round trains all models on about 4 million rows, which takes time. Three rounds give 30 test weeks, and I
added a corrected test and a bootstrap to be careful with this small number.

**Q: How did you choose the model settings (hyperparameters)?**
A: In each round, I tried a few settings on the last 10 training weeks and kept the best. The test weeks were never used.

**Q: What is early stopping?**
A: The model adds trees one by one and stops when it no longer improves on the check weeks. *Example:* like stopping
study when your practice score stops going up.

**Q: Did you fix the random seed?**
A: Yes, 42, so the results are the same every time the notebooks are run.

---

## 7. The models

**Q: Which models did you use?**
A: Three groups. Simple: copy last week, 26-week average, Linear Regression. Tree models: Random Forest, XGBoost,
LightGBM. Quantile models: Linear Quantile Regression, quantile XGBoost, quantile LightGBM.

**Q: Why these simple models?**
A: They are a fair starting point. If a smart model cannot beat "copy last week", it is not worth using.

**Q: What is a tree model, simply?**
A: A set of yes/no questions. *Example:* "Is there an offer? Yes. Is it a drink? Yes. Then expect about 400 orders."

**Q: What is the difference between Random Forest and boosting (XGBoost, LightGBM)?**
A: Random Forest builds many trees separately and averages them, like asking 100 friends and taking the average.
Boosting builds trees one after another, and each new tree fixes the mistakes of the earlier ones.

**Q: Why tree models?**
A: They are the strongest on table data, handle many kinds of inputs, need little preparation, and are fast on
millions of rows. LightGBM won the M5 contest.

**Q: Why not deep learning (LSTM)?**
A: Research shows tree models still beat deep learning on table data. Deep learning is slower and harder to set up,
and I had 12 weeks. It is future work.

**Q: Why not ARIMA or other classic time-series models?**
A: They build one model per series and cannot easily use price and offers. With 3,597 short, gappy series, one
global model with these inputs is a better fit. My simple baselines play the role of the classic methods.

**Q: Is CatBoost deep learning?**
A: No. CatBoost is a tree model, in the same family as XGBoost and LightGBM.

**Q: What is a quantile model, in one sentence?**
A: A normal model says "about 100 meals"; a quantile model says "130 meals will be enough in 8 out of 10 weeks",
and you choose how safe you want to be.

**Q: Why quantile levels 0.3 to 0.9?**
A: Each level matches a cost ratio. If a missed order costs half a wasted meal, the best level is 0.33; if it costs
5 times more, it is 0.83. So 0.3 to 0.9 covers every case I test.

**Q: What settings did the final models use?**
A: LightGBM: learning rate 0.05, 63 or 255 leaves, about 110–250 trees. XGBoost: depth 8, about 120 trees.
Random Forest: 100 trees, depth 16, trained on 1 million random rows.

**Q: Why was Random Forest trained on a sample?**
A: It is slow and uses a lot of memory on 4 million rows. One million random rows is enough for it to learn the patterns.

**Q: What is "crossing", and how did you handle it?**
A: Sometimes the level-0.7 answer is bigger than the level-0.8 answer, which makes no sense. It happened in about
1 of 4 forecasts of quantile LightGBM. I sort the answers so they always go up. Quantile XGBoost had no crossing.

---

## 8. The scores (metrics)

**Q: Which scores did you use?**
A: WAPE, MAE, RMSE and RMSLE for accuracy; bias and over-forecast % for the direction of the error; pinball loss
and coverage for the quantile models; and total cost for the decision.

**Q: What is WAPE, with an example?**
A: Total error divided by total real orders. *Example:* real orders 100 and 200 (total 300), errors 10 and 50
(total 60): WAPE = 60 ÷ 300 = 20%. So "wrong by 20 meals in every 100".

**Q: Why WAPE and not MAPE?**
A: MAPE gives huge numbers for small weeks. *Example:* guessing 30 when it was 15 is a 100% error. WAPE adds
everything up first, so small weeks do not take over.

**Q: What is the difference between MAE and RMSE?**
A: Both are the average error in meals. RMSE punishes big mistakes more. *Example:* errors of 10 and 50 give
MAE 30 but RMSE 36.

**Q: What is RMSLE?**
A: The error on the log scale, so it works in "percent-like" terms. Guessing 20 when it was 10 counts about the same as
guessing 2,000 when it was 1,000.

**Q: What is bias?**
A: The average of (forecast − real). *Example:* bias −22 means the forecast is 22 meals too low on average.

**Q: What is pinball loss?**
A: The score for a quantile forecast. For a high level, being too low is punished more. *Example:* at level 0.8,
being too low costs 4 times more than being too high.

**Q: What is coverage?**
A: How often the real orders were at or below the forecast. A level-0.8 forecast should have coverage near 80%.

**Q: Why did you not report "accuracy" as a percent, or R²?**
A: "Accuracy" is for yes/no problems, not for numbers. R² can look very high just because big centres sell more
than small ones. WAPE, MAE and RMSE are the standard scores in forecasting. As a quick extra check only, the R²
of LightGBM is about 0.82.

---

## 9. Results: accuracy (RQ1)

**Q: Which model is the most accurate?**
A: LightGBM and XGBoost, in a tie: WAPE about 28.6%, MAE 71, RMSE 152. Random Forest is next (29.4%), then Linear
Regression (32.0%), the 26-week average (38.4%) and copy last week (50.9%).

**Q: What does 28.6% mean in practice?**
A: If all the real orders add up to 1,000 meals, the forecasts are wrong by about 286 meals in total.

**Q: How much better is LightGBM than the best simple model?**
A: Its error is 10.5% smaller than Linear Regression's, and 44% smaller than copy last week's.

**Q: Is the result the same in every round?**
A: Yes. The order of the models is the same in all three rounds. Round 2 is the hardest and round 3 the easiest.

**Q: Does the error grow further ahead?**
A: A little. LightGBM is wrong by about 27 in 100 for next week and about 32 in 100 for ten weeks ahead.

**Q: Why does "copy last week" get so bad after one week?**
A: Orders jump a lot from week to week, mostly because of offers. So one single week says little about ten weeks later.

**Q: Why are LightGBM and XGBoost the same?**
A: They are the same idea (boosted trees) with small technical differences, and they use the same inputs. The test
gives p = 0.96, so there is no real difference.

**Q: Why are all models too low on average?**
A: They learn on the log scale, which aims for the "normal" week, not the average. *Example:* weeks of 100, 100, 100
and 500 have a normal value of 100 but an average of 200. Guesses near 100 miss the big weeks.

---

## 10. Results: quantile models (RQ2)

**Q: Do the quantile levels keep their promise?**
A: Yes. For level 0.8, the forecast was enough in 77% (quantile LightGBM), 78% (quantile XGBoost) and 80%
(Linear QR) of cases. The tree models are 2–3 points under the target at high levels.

**Q: What about the normal forecast?**
A: It is enough in only 46% of cases, and you cannot change that, because it gives one number only.

**Q: Which quantile model is best?**
A: The two tree models, in a tie (average pinball loss 30.4). Linear QR is clearly worse (33.8).

**Q: Linear QR has the best coverage. Why is it not the best model?**
A: Coverage only checks "how often enough". Pinball loss also checks "by how much". Linear QR is right on average but
less exact for each meal, so its orders are less efficient.

**Q: Is the middle level (0.5) a good normal forecast?**
A: Yes. Quantile LightGBM at 0.5 has WAPE 28.1% and bias −15, slightly better than the normal LightGBM (28.6%, −22).

---

## 11. Results: waste, shortage and cost (RQ3)

**Q: How did you work out waste and shortage without real data?**
A: With simple rules. The kitchen makes what the forecast says. Meals not sold are wasted. Orders that cannot be
served are lost. *Example:* make 100, sell 80 → 20 wasted. Make 100, 120 wanted → 20 missed.

**Q: What is the cost ratio?**
A: How many times more a missed order costs than a wasted meal. I test 0.5, 1, 1.5, 2, 3, 4 and 5, because the real
value is unknown.

**Q: Which way of ordering is cheapest?**
A: Quantile LightGBM or quantile XGBoost, ordered at the level from the newsvendor rule. They are the cheapest at
every cost ratio.

**Q: How much do they save?**
A: Against the normal LightGBM forecast: 2% at ratio 1, 14% at ratio 2, 25% at ratio 3, and 40% at ratio 5.
*Example:* at ratio 3, a cost of £100 becomes about £75.

**Q: Why is the saving small at ratio 1?**
A: When both mistakes cost the same, the best order is the middle (level 0.5), and the normal forecast is already
close to the middle.

**Q: What happens to waste and shortage?**
A: The normal forecast always wastes about 10 and misses about 19 in every 100 meals. At ratio 5, the quantile
forecast wastes more (32 in 100) but misses far fewer (6 in 100). You trade some waste for fewer missed orders.

**Q: So does your method increase food waste?**
A: Only when missed orders are more expensive than waste, because then that is the cheaper choice. When waste is
the bigger problem (ratio 0.5), it orders less and waste falls from 10 to 6 in every 100.

**Q: Does the newsvendor rule work on this data?**
A: Yes. I tried every level, and the cheapest level was the same as the rule or the nearest level I tested.

**Q: What is the safety stock, and why did you test it?**
A: Real planners add "a little extra just in case". I built this rival: the normal forecast plus an extra amount
learned from its own past mistakes. It is a fairer comparison than the plain forecast.

**Q: Does the quantile model still win against the safety stock?**
A: Yes, by 3% to 7% at every cost ratio. *Why:* the safety stock adds the same share to every meal, but the quantile
model adds more only for jumpy meals.

---

## 12. Statistics: is it real or luck?

**Q: How do you know the differences are real?**
A: I used the Diebold–Mariano test. A p-value below 0.05 means the difference is very unlikely to be luck. The tree
models against Linear Regression, and the quantile models against the normal forecast, all have p below 0.001.

**Q: What is the "corrected" version of the test?**
A: The simple test treats every week as separate. But a busy week is often followed by another busy week, and I have
only 30 test weeks. The corrected version (Newey–West and Harvey–Leybourne–Newbold) allows for both. All main
results stayed real.

**Q: What is the bootstrap, and what did it show?**
A: I picked the 3,597 meal–centre pairs again at random, 1,000 times, and worked out the saving each time. This gives
a range. *Example:* at ratio 3 the saving is 25.0%, with a range of 24.1% to 25.8%. Every range is above zero.

**Q: Is 30 test weeks enough?**
A: It is small, which is why I used the corrected test and the bootstrap over 3,597 series. Each test week also
contains thousands of forecasts.

---

## 13. Hard and critical questions

**Q: Other papers report 99% accuracy. Yours is much lower. Why?**
A: Those numbers usually come from an easier or unfair test: a random split (the model sees the future), a test on
the training data, a forecast only one step ahead, or a different score such as R². On the same data and the same
test weeks as the published study, my errors are about 40% smaller.

**Q: Your RMSLE (0.48) is higher than Panda & Mohanty's (0.29). Is your model worse?**
A: On the same test weeks my RMSE and MAE are about 40% lower. Their paper does not fully explain how the inputs were
built, and some of its numbers do not match each other, so that one score is hard to compare.

**Q: Is an error of 29 in 100 good enough to be useful?**
A: Yes. It is for one meal at one centre, up to 10 weeks ahead, which is always much harder than a company total.
And the main result is about the order: even with this error, the right quantile level cuts the cost by up to 40%.

**Q: The costs are not real. So is the saving meaningless?**
A: The exact percent depends on the real costs. But the pattern is clear and holds for all seven ratios: the more a
missed order costs, the more a quantile model saves. A company can put in its own costs and get its own number.

**Q: Could the model be improved further?**
A: A little, maybe. But even for next week, with the newest information, the error is 26.5%. This shows most of
the remaining error is sudden change that the data cannot explain, not a weak model.

**Q: Is your comparison with the normal forecast fair? Nobody orders exactly the forecast.**
A: That is why I added the safety-stock rival. The quantile models still win by 3–7%.

**Q: Did you choose the best quantile level by looking at the test results?**
A: No. The level comes from the newsvendor rule before any testing. Checking all levels afterwards only confirmed the rule.

**Q: Why should we trust simulated results?**
A: The rules are simple, written down, and the same for every model. The forecasts are real, tested fairly, and the
code is public. Only the prices are assumed, and I test a wide range of them.

**Q: Is using the target week's price a kind of leakage?**
A: No. It is planned information, known before the order is made. Leakage would be using the real orders of future weeks.

**Q: Why is XGBoost sometimes cheaper than LightGBM in the cost table?**
A: The difference is less than one point and goes both ways (LightGBM is cheaper at low ratios, XGBoost at high ones).
They are equal in practice.

---

## 14. Limitations

**Q: What are the limitations of your study?**
A: (1) Costs are estimated, not real. (2) Only one dataset. (3) Simple rules: each week on its own, no stock kept,
no customer coming back. (4) Recorded orders are treated as true demand. (5) Missing weeks are skipped.
(6) Only 30 test weeks. (7) No deep learning. (8) Old, anonymous data, so no holidays or weather.

**Q: What is the biggest weakness?**
A: The costs are not real, and there is only one dataset.

**Q: What is "hidden demand" (censored demand)?**
A: If food ran out, some people who wanted a meal were never recorded. *Example:* 120 people wanted it, 100 meals
were made, and the data may show 100. So real demand may be a bit higher than the data says.

**Q: Would the results hold for another company?**
A: The method works anywhere. The exact numbers may change, so testing on more datasets is future work.

---

## 15. Using the results in real life

**Q: What should a company do with your results?**
A: Three steps. (1) Estimate how much a missed order costs compared with a wasted meal. (2) Find the matching level
(for example ratio 3 → level 0.75). (3) Order the quantile forecast at that level.

**Q: What if the company does not know its cost ratio?**
A: Even a rough guess helps ("a missed order is about 2 to 4 times worse"). The savings are positive for the whole
range, and the app lets a planner try different values.

**Q: What does the Streamlit app do?**
A: It shows the data and the results in simple words, and it has an order planner: choose the cost ratio and see
the suggested level and the expected waste and shortage.

**Q: Is the system ready for real use (deployment)?**
A: No. It is a research prototype. A real system would need live data, real costs and regular retraining.

**Q: How long does training take?**
A: On a normal laptop: about 30–45 minutes for the tree models and about 70 minutes for the quantile models.

---

## 16. Ethics and repeating the work

**Q: Are there any ethical issues?**
A: No personal data is used. The data is public and anonymous, and it is used only for academic research.

**Q: Can someone repeat your work?**
A: Yes. The data is public, all code is in this repository, each step has its own notebook, the random seed is fixed,
and the README explains how to set up and run everything.

**Q: Which tools did you use?**
A: Python with Jupyter notebooks; pandas and NumPy for data; Matplotlib for charts; scikit-learn, XGBoost and
LightGBM for models; statsmodels for Linear Quantile Regression; Streamlit for the app; Git for versions.

---

## 17. Future work

**Q: What would you do next?**
A: Use real cost data from a company; test more datasets; try deep learning; allow food to be kept for more than
one week; handle hidden demand when food runs out; and train the model directly on the cost.

**Q: How would real cost data change the study?**
A: Instead of a range of ratios, there would be one real ratio for each meal, and the saving would be real money.

**Q: What does "allow food to be kept" mean?**
A: Some ingredients last more than one week. Then waste this week is smaller, and the best order depends on what is
already in stock. This is a "multi-week" stock problem.

---

## 18. Personal questions

**Q: What was the hardest part?**
A: Building the inputs without any data leakage, and making one model that works for all 10 weeks ahead.

**Q: What did you learn?**
A: That a good score is not the same as a good decision; that fair testing matters more than a clever model; and
that simple rivals and honest limits make research stronger.

**Q: What surprised you?**
A: Two things. Last week's orders are almost useless for forecasting far ahead. And a simple "add a little extra"
trick gets most of the saving, but the quantile model is still better.

**Q: If you started again, what would you change?**
A: I would try to get real cost data from a company at the start, and add a second dataset.

**Q: Why should this be published?**
A: It answers a practical question with public data and open code, it uses fair tests, and it shows clearly when
and why quantile forecasts save money for fresh food.
