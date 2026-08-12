We assume the next days the stock price will be high. we can say this is a hypothetical assume prediction. it comes from the previous knowledge and it will not correct prediction every time. we fix the market value in overall idea. after experiment we update our hypothesis. lets start with an example and procced further among some experiment agaist the hypothesis. suppose we say tomorrow the stock price will be high.  

I think we have to first talk the traditional reason. when the stock price goes higher it's highly consistance in few days. unless the goverment new rules apply or accidently something happend in the world. suddendly peek performance of stock ignore by the investor or academic researcher. they first observe the market behaviour and understand the pattern of performance. so, itconfirm nobody affect the stock matket and it goes further in consistant way.  

now observe the financial transfer to the market. we have to observe how it effect the market value and what is the step or process for contribution, debug each step carefully, trace the underlying relation. investor don't invest the money single time instead of they apply it consist way. first invest then obeserve and second invest and continue.   

Features needed
- 5-day return
- 20-day return
- 60-day return
- Rolling average
- Momentum indicator

Models to test
- Linear Regression
- LightGBM
- CatBoost

Evaluation plan
1. Train a baseline model.
2. Add momentum features.
3. Compare validation metrics.
4. Analyze feature importance.
5. Measure portfolio performance.

Decision criteria
Accept if momentum features consistently improve
validation performance and risk-adjusted returns
across multiple validation periods.