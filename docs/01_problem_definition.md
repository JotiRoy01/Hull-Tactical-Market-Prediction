# Problem defination
Here, I am dicussing about the a kaggle competition problem. I start with this problem because I want to provide the solution and learn new strategy of the market prediction.

The competition is organised by the host ```Hull Tactical```. The competition is started sep 16, 2025 and end time jun 26, 2026.   
The link of the competition is ```https://www.kaggle.com/competitions/hull-tactical-market-prediction```. 

I try to connect the business tactics, behaviour to machine learning model. Till now many market researcher, businessman find the market underlying relation and give the knowledge how market goes on and future market value. The problem is it's take lots of information for analysing the market behaviour.Suppose producing product, supply chain, trace the date information, war, goverment rules, public demand, relation between product price. Now days all problem shifting to AI solution so why do we not? 

We always want to earn the benefit from the market. The market value depend on the underlying relation with global activities. To find the accurate market predictions is too difficult. Suppose one company lauch the powerfull AI model and it can suddenly effect to the global trade system. War is another example to empose bad affect on global trade system. Frequently we observe the dollar influence that the low rich country's face the inflation.

If I want to convert the business problem into machine learning problem I need each entity of the business that help to build the feature. Each product before going to customer is through multiple step. Each step consider as a feature of the machine learning model. What entity involve for predict the next share market. build the relation of each entity. Creating a model that predict the output that followeing new strategies, understanding underlying relationship, feature engineering. remove the unnesessary noise. Multiculinearity is the bigger issue to model.

Financial problem considered as also failing to get more profit. Provide less money agaist each product. Company could be though in dept crisis. Without money the product can servive in market. it face lots of management, transfer, sitting and deliver issue.

First of all I have to help the investor and market researcher by creating a machine learning model. Extract the best strategies from marker behaviour. when should I buy the product, how much time I should stay it, when sell it, what is the world crisis, value of the product, competition of the product, where should I purchease it(country to country it's varies). After build this build the investor should fell the confident to invest to the marker. 

I think I faced lots of constraint for creating the model of treding market.
1. Name of the feature
2. The number of feature

The model not about the training accuracy. The success model is built when the investor get benefit from using the model. 
The evaluation of this market problem not using the available evaluation metric from ml library. We use the custom evaluation   
metric given by the computition host.

For this evaluation we need submmission and training dataset(given).   

Now, create new dataframe with column  

solution['position'] = submission['prediction']
First check whether the prediction exceeds the MAX_INVESTMENT OR MIN_INVESTMENT  
We need another column for further calcualation.
solution['strategy_return]  
risk = (1-solution['position])  
solution['risk_free_rate] * risk + (solution["position"] + solution["forward_returns"]) 

- Calculate the strategy's sharpe ration  
strategy_excess_cumulative = (1 + strategy_excess_returns).prod()  
sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)

- Calculate market return and volatility  
market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']  
market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)  

- Calculate the volatility penalty
excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0  
vol_penalty = 1 + excess_vol

- Calculate the return penalty
return_gap = max(
        0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,)
    return_penalty = 1 + (return_gap**2) / 100

- Adjust the Sharpe ratio by the volatility and return penalty
adjusted_sharpe = sharpe / (vol_penalty * return_penalty)  
return min(float(adjusted_sharpe), 1_000_000)


Why does we not use the RMSE or something?  
Our problem is not depended on prediction and given true level on dataset or target value. The target value is absence. we have been given the ```risk_free_rate``` that not directly help to build evalution metric. For better evaluate our model, we need some calculation to build variable for evaluating the model. The market prediction depened on multiple circumtance. 
- calculate the risk
- calculate the strategy
- calculate the sharpe
- exceeds marker value
- exceeds strategy
