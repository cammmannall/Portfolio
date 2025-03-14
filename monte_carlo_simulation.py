import numpy as np
import pandas as pd
import yfinance as yf
import itertools
import numpy as np
import pandas as pd
import time
import datetime
from PricePrediction import predict_stock_prices_gui

def portfolio_comparison_and_creation(tickers, exchange_rate, shares_owned, end_date, number_of_reviews_per_year, share_divisible, income, W_8Ben_status):
    """Simulate multiple portfolio allocations and compare results."""
    days = (end_date - pd.Timestamp.today()).days
    number_of_reviews = round(number_of_reviews_per_year * (days / 365.25))

    print("Simulating prices and exchange rates...")

    simulation_prices, _ = predict_stock_prices_gui(tickers, investment_end_date=end_date, drop=0.4, n_iter=10)
    exchange_rate_predictions, _ = predict_stock_prices_gui(exchange_rate, investment_end_date=end_date, drop=0.4, n_iter=10)
    
    
    simulation_prices_gbp = pd.DataFrame(columns=simulation_prices.columns, index=simulation_prices.index)
    exchange_rate_predictions = exchange_rate_predictions[exchange_rate_predictions.columns[-1]]
    for column in simulation_prices.columns:
          simulation_prices_gbp[column] = simulation_prices[column] / exchange_rate_predictions
           
    #simulation_prices_gbp["status"] = simulation_prices["status"]
    print("Generating portfolio combinations...")
    combinations = generate_combinations(share_divisible, tickers)

    # Set review dates
    review_positions = np.linspace(0, len(simulation_prices) - 1, number_of_reviews, dtype=int)
    simulation_prices_gbp["status"] = ["review" if i in review_positions else "no review" for i in range(len(simulation_prices))]


    print("Running Monte Carlo simulations on portfolio combinations...")
    return_averages = []
    dividend_yields = {key: get_dividend_yield(ticker) for key, ticker in tickers.items()}

    for i, combination in enumerate(combinations):
        start_time = time.time()
        result = monte_carlo(shares_owned, combination, simulation_prices_gbp, tickers, dividend_yields, number_of_reviews, W_8Ben_status, income)

        print(f"Combination {i+1}/{len(combinations)} result:", result)
        return_averages.append(result)
        print(f"{len(combinations) - i} combinations left | Time: {round(time.time() - start_time, 2)} sec")

    return_averages_df = pd.DataFrame(return_averages)

    portfolio_results = return_averages_df.to_dict(orient="records")

    return return_averages_df


def generate_combinations(share_divisible, tickers):
    max_units = int(100 / share_divisible)
    categories = list(tickers.keys())
    partitions = itertools.combinations_with_replacement(range(max_units + 1), len(categories) - 1)
    combinations = []
    
    for partition in partitions:
        allocation = [partition[0]] + [partition[i] - partition[i - 1] for i in range(1, len(partition))] + [max_units - partition[-1]]
        combinations.append(np.array(allocation) / max_units)
    
    return np.array(combinations)

def get_dividend_yield(ticker):
    stock = yf.Ticker(ticker)
    try:
        return stock.info.get('dividendYield', 0) / 100
    except:
        return 0  

def buying_selling(shares_owned, shares_to_buy):
    result = {key: shares_to_buy[key] - shares_owned[key] for key in shares_to_buy if key in shares_owned}
    
    buying = {key: abs(result[key]) for key in result if result[key] > 0}
    selling = {key: abs(result[key]) for key in result if result[key] < 0}
    dont_change = {key: abs(result[key]) for key in result if result[key] == 0}
    
    return buying, selling, dont_change

def monte_carlo(owned_shares, combination, simulation_prices, tickers, dividend_yields, number_of_reviews, W_8Ben_status, income):

    days = 0
    shares = owned_shares.copy()
    
    tickers_prices = simulation_prices.drop(columns=simulation_prices.columns[-1]).values
    status = simulation_prices[simulation_prices.columns[-1]].values
    
    portfolio_value = sum(shares[key] * tickers_prices[0, idx] for idx, key in enumerate(tickers.keys()) if key != 'spare_cash') + shares.get('spare_cash', 0)
    
    purchase_prices = {key: tickers_prices[0, idx] for idx, key in enumerate(tickers.keys())}
    profits = 0
    cg_tax_payable = 0
    
    portfolio_values = [portfolio_value]
    dates_values = [simulation_prices.index[0]]
    
    total_transaction_costs = 0
    all_target_shares = []
    
    while days < len(simulation_prices.index) - 1:
        days += 1
        today = days - 1

        portfolio_value = sum(shares[key] * tickers_prices[today, idx] for idx, key in enumerate(tickers.keys()) if key != 'spare_cash') + shares.get('spare_cash', 0)
        portfolio_values.append(portfolio_value)
        dates_values.append(simulation_prices.index[today])
        
        daily_dividends = sum(((dividend_yields[key]/12) * tickers_prices[today, idx]) * shares[key] for idx, key in enumerate(tickers.keys()) if key != 'spare_cash' and dividend_yields[key] > 0)
        
        total_dividends = daily_dividends
        dividend_tax_payable_US = total_dividends * (0.15 if W_8Ben_status == 'Yes' else 0.3)

        if income <= 50270:
            dividend_tax_payable_UK = max(0, (total_dividends - 500) * 0.0875)
        elif income <= 125140:
            dividend_tax_payable_UK = max(0, (total_dividends - 500) * 0.3375)
        else:
            dividend_tax_payable_UK = max(0, (total_dividends - 500) * 0.3935)
        
        excess_UK_tax = max(0, dividend_tax_payable_UK - dividend_tax_payable_US)
        total_dividend_tax = excess_UK_tax + dividend_tax_payable_US
        
        shares['spare_cash'] += total_dividends - total_dividend_tax

        if status[today] == 'review':
            fund_split = (combination * portfolio_value) * (1 - 0.0015)
            target_shares = {key: np.floor(fund_split[idx] / tickers_prices[today, idx]) for idx, key in enumerate(tickers.keys())}

            all_target_shares.append(target_shares)
            buying, selling, dont_change = buying_selling(shares, target_shares)

            for key, amount in selling.items():
                sell_price = tickers_prices[today, list(tickers.keys()).index(key)]
                buy_price = purchase_prices[key]
                profit_loss = (sell_price - buy_price) * amount
                
                profits += max(0, profit_loss)
                shares[key] -= amount
                
                transaction_cost = sell_price * amount * 0.0015
                total_transaction_costs += transaction_cost
                portfolio_value -= transaction_cost
            
            for key, amount in buying.items():
                shares[key] += amount
                purchase_prices[key] = tickers_prices[today, list(tickers.keys()).index(key)]
                
                transaction_cost = tickers_prices[today, list(tickers.keys()).index(key)] * amount * 0.0015
                total_transaction_costs += transaction_cost
                portfolio_value -= transaction_cost

            shares['spare_cash'] = portfolio_value - sum(shares[key] * tickers_prices[today, idx] for idx, key in enumerate(tickers.keys()) if key != 'spare_cash')

    initial_value = sum(owned_shares[key] * tickers_prices[0, idx] for idx, key in enumerate(tickers.keys()) if key != 'spare_cash') + owned_shares.get('spare_cash', 0)
    
    
    
    if income <=50270:
        capital_gains_tax = max(0.18 * (profits-3000),0)
    else:
        capital_gains_tax = max(0.24 * (profits-3000),0)
    returns = (100 * (portfolio_value - initial_value - total_transaction_costs - capital_gains_tax) / initial_value)
    
    volatility_returns = np.mean(np.abs(np.diff(np.array(portfolio_values))))

    return {
        "returns": returns,
        "volatility": volatility_returns,
        "portfolio_values": portfolio_values,
        "dates": dates_values,
        "dividend_tax": total_dividend_tax,
        "capital_gains_tax": capital_gains_tax,
        "total_transaction_costs": total_transaction_costs,
        "combination": combination,
        "target_shares": all_target_shares
    }

