import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import csv
import os
import datetime
from PricePrediction import predict_stock_prices_gui
from monte_carlo_simulation import portfolio_comparison_and_creation
import matplotlib.pyplot as plt
import time
DIVIDENDS_COLUMNS = ["username", "Consumer Staples", "Healthcare", "Tech", "Bonds", "Yen to USD", "Gold", "spare_cash"]


# Load DividendsApp.csv
def load_dividends_data():
    if os.path.exists("DividendsApp.csv"):
        return pd.read_csv("DividendsApp.csv")
    else:
        return pd.DataFrame(columns=DIVIDENDS_COLUMNS)

def save_dividends_data(dividends_df):
    dividends_df.to_csv("DividendsApp.csv", index=False)

def add_new_user_to_dividends(username):
    dividends_df = load_dividends_data()
    
    if username in dividends_df["username"].values:
        return

    new_user = pd.DataFrame([[username, 0, 0, 0, 0, 0, 0, 10000000]], columns=DIVIDENDS_COLUMNS)
    dividends_df = pd.concat([dividends_df, new_user], ignore_index=True)
    save_dividends_data(dividends_df)

def load_user_data():
    if os.path.exists("user_data1.csv"):
        with open("user_data1.csv", mode="r") as file:
            reader = csv.DictReader(file)
            return {row["username"].strip().lower(): row for row in reader}
    return {}

def save_user_data(user_data):
    fieldnames = ["username", "age", "kids", "house", "married", "income", "W-8Ben", "end date", "risk level"]
    with open("user_data1.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for user in user_data.values():
            writer.writerow(user)

# Risk Model Function
def risk_model(age, kids, house, married):
    if age < 18:
        return None
    if age > 95:
        age = 95

    if age in [18, 19]:
        age = 20

    if age < 62:
        age_factor = -0.0002272817 * (age**2) + 0.0260646383 * age + 0.0234075396
    else:
        age_factor = 0.0002703257 * (age**2) - 0.0414619493 * age + 2.3322452768

    weight_age = 0.37327153
    weight_kids = -0.39697758
    weight_house = 0.06486365
    weight_married = 0.16488725
    data_min = -1.87736
    data_max = 0.6925303

    risk = weight_age * age_factor + weight_kids * kids + weight_house * house + weight_married * married
    normalized_risk = np.round(100 * ((risk - data_min) / (data_max - data_min)), 2)

    return max(0, normalized_risk)

# ---- Streamlit Setup ----
st.set_page_config(page_title="Investment Portfolio Finder", layout="wide")
page = st.sidebar.selectbox("Select a Page", ["Login", "Portfolio Finder", "Risk Survey"])

# ---- Login Page ----
if page == "Login":
    st.title("🔑 User Login")
    username = st.text_input("Enter your username:")
    user_data = load_user_data()

    if st.button("Login"):
        username = username.strip().lower()
        
        if username in user_data:
            st.success(f"✅ Welcome back, {username.capitalize()}!")
            user = user_data[username]
            st.session_state["user"] = user
            st.session_state["is_logged_in"] = True
            st.rerun()
        
        else:
            new_user = {
                "username": username,
                "age": "18",
                "kids": "0",
                "house": "0",
                "married": "0",
                "income": "10000",
                "W-8Ben": "0",
                "end date": (datetime.datetime.now() + datetime.timedelta(weeks=52*5)).strftime("%Y-%m-%d"),
                "risk level": "50"
            }
            user_data[username] = new_user
            save_user_data(user_data)
            add_new_user_to_dividends(username)
            st.session_state["user"] = new_user
            st.session_state["is_logged_in"] = True
            st.warning("No previous data found. Registering as a new user...")
            time.sleep(2)
            st.success("✅ Registration Complete. You are now logged in!")
            time.sleep(2)
            st.rerun()



elif page == "Portfolio Finder":
    if "is_logged_in" not in st.session_state or not st.session_state["is_logged_in"]:
        st.warning("🔒 Please log in first.")
        st.stop()

    st.title("💰 Investment Portfolio Finder")

    user = st.session_state["user"]
    username = user["username"]
    risk_saved = int(user["risk level"])

    dividends_df = load_dividends_data()

    if username not in dividends_df["username"].values:
        st.warning("⚠️ No portfolio found. Creating a new portfolio for you...")
        
        new_portfolio = pd.DataFrame([[username, 0, 0, 0, 0, 0, 0, 10000000]], columns=DIVIDENDS_COLUMNS)
        dividends_df = pd.concat([dividends_df, new_portfolio], ignore_index=True)
        save_dividends_data(dividends_df)

    user_dividends = dividends_df[dividends_df["username"] == username]

    st.header("Investor Profile")
    risk_level = st.slider("Select Risk Level (0: Low, 100: High)", 0, 100, risk_saved)

    income_options = {
        "Less than £50,270": 50270,
        "Between £50,271 - £125,140": 125140,
        "More than £125,140": 125141
    }
    income_category = st.selectbox("Select Your Yearly Income Bracket", list(income_options.keys()))
    income = income_options[income_category]

    investment_years = st.slider("Investment Duration (years)", 1, 10, 5)
    W_8Ben_status = st.radio("Do you have a W-8BEN form in the last 3 years?", ["Yes", "No"])

    st.subheader("📊 Your Current Portfolio")
    st.write(user_dividends)

    if st.button("Find Best Portfolio"):
        user["risk level"] = str(risk_level)
        save_user_data({username: user})
        st.session_state["user"] = user
        
        st.write("🔄 Running Monte Carlo simulations...")

        tickers = {
            'Consumer Staples': 'VDC',
            'Healthcare': 'VHT',
            'Tech': 'VGT',
            'Bonds': 'TLT',
            'Yen to USD': 'JPY=X',
            'Gold': 'GLD'
        }

        exchange_rate = {'GBP to USD': 'GBPUSD=X'}
        
        share_divisible = 10
        end_date = pd.to_datetime(datetime.datetime.now() + datetime.timedelta(weeks=52 * investment_years))
        initial_cash = 10000000
        W_8Ben_status = "Yes"

        return_averages = portfolio_comparison_and_creation(
            tickers, exchange_rate, user_dividends.iloc[:, 1:].to_dict(orient="records")[0],
            end_date, number_of_reviews_per_year=1, share_divisible=share_divisible,
            income=income, W_8Ben_status=W_8Ben_status  # Pass the selected value
        )


        if return_averages.empty:
            st.error("❌ Error: No portfolio data generated.")
            st.stop()

        return_averages["percentile"] = (return_averages["volatility"].rank(pct=True) * 100).round(0).astype(int)

        filtered_portfolios = return_averages[
            return_averages["percentile"] == risk_level
        ]

        if not filtered_portfolios.empty:
            best_portfolio = filtered_portfolios.loc[filtered_portfolios["returns"].idxmax()]
        else:

            best_portfolio = return_averages.loc[return_averages["volatility"].idxmin()]

        print("✅ Best Portfolio Selected:")
        print(best_portfolio)

        optimized_weights = best_portfolio["combination"]
        optimized_weights = optimized_weights / np.sum(optimized_weights)


        expected_return = best_portfolio["returns"] / 100

        volatility = (best_portfolio["volatility"] / initial_cash) * 100 

        future_value = (expected_return * initial_cash) + initial_cash 

        confidence_factor = 1.96 

        annual_volatility = volatility / np.sqrt(investment_years) 

        compounded_volatility = (1 + (annual_volatility / 100)) ** investment_years 

        lower_bound = initial_cash * ((1 + expected_return) / compounded_volatility - confidence_factor * (annual_volatility / 100))
        upper_bound = initial_cash * ((1 + expected_return) * compounded_volatility + confidence_factor * (annual_volatility / 100))

        lower_bound = max(0, round(lower_bound, 2))
        upper_bound = round(upper_bound, 2)

        annual_return = (((future_value-initial_cash) / initial_cash) / (investment_years)) * 100 

        print(f"Expected Return: {expected_return * 100:.2f}%")
        print(f"Annualized Volatility: {annual_volatility:.2f}%")
        print(f"Worst-Case Compounded Volatility Factor: {compounded_volatility:.4f}")
        print(f"Portfolio Value Estimate: ${future_value:,.2f}")
        print(f"Confidence Interval: ${lower_bound:,.2f} - ${upper_bound:,.2f}")

        st.subheader(f"📈 Projected Portfolio Value in {investment_years} Years on {(datetime.datetime.now()+datetime.timedelta(weeks = 52*investment_years)).date().strftime('%d-%m-%Y')}")
        st.write(f"**Estimated Portfolio Value:** £{future_value:,.2f}")
        st.write(f"**95% Confidence Interval:** £{lower_bound:,.2f} - £{upper_bound:,.2f}")
        st.write(f"**Average Annual Return Estimate:** {annual_return:.2f}% per year")
        st.write(f"Come back in 1 year for rebalancing on {(datetime.datetime.now()+datetime.timedelta(weeks = 52)).date().strftime('%d-%m-%Y')}")

        allocations = {asset: optimized_weights[i] * initial_cash for i, asset in enumerate(tickers.keys())}
        current_exchange_rate = yf.Ticker("GBPUSD=X").history(period="1d")["Close"][-1]
        stock_prices = {}
        for asset, ticker in tickers.items():
            try:
                stock_data = yf.Ticker(ticker).history(period="1d")["Close"]
                stock_prices[asset] = stock_data.values[-1]/current_exchange_rate if not stock_data.empty else None
            except Exception as e:
                print(f"⚠️ Error fetching price for {ticker}: {e}")
                stock_prices[asset] = None

        shares_to_buy = {asset: (allocations[asset] // stock_prices[asset]) if stock_prices[asset] else 0 for asset in tickers.keys()}

        for asset in tickers.keys():
            dividends_df.loc[dividends_df["username"] == username, asset] = shares_to_buy[asset]

        remaining_cash = initial_cash - sum(shares_to_buy[asset] * stock_prices[asset] if stock_prices[asset] else 0 for asset in tickers.keys())
        dividends_df.loc[dividends_df["username"] == username, "spare_cash"] = round(remaining_cash, 2)

        save_dividends_data(dividends_df)
        st.subheader("📊 Updated Portfolio After Optimisation")
        st.write(dividends_df[dividends_df["username"] == username])

        labels = list(tickers.keys())
        combination_weights = best_portfolio["combination"] 
        normalized_weights = combination_weights / np.sum(combination_weights) 


        filtered_data_pie = [(label, weight) for label, weight in zip(labels, normalized_weights) if weight > 0]

        if filtered_data_pie:
            labels, sizes = zip(*filtered_data_pie)
            sizes = [size * 100 for size in sizes]  
        else:
            labels, sizes = [], []

        import plotly.graph_objects as go
        import streamlit as st

        purchase_df = pd.DataFrame({
            "Ticker": labels,
            "Weight (%)": sizes,
            "Shares Purchased": [shares_to_buy[asset] for asset in labels],
            "Total Invested (£)": [round(shares_to_buy[asset] * stock_prices[asset] if stock_prices[asset] else 0, 2) for asset in labels]

        })
        st.write(purchase_df)

        st.subheader("📌 Portfolio Allocation")

        fig_pie = go.Figure(
            data=[go.Pie(
                labels=labels,
                values=sizes,
                hole=0,
                textinfo="label+percent"
            )]
        )

        fig_pie.update_layout(
            title="Portfolio Breakdown",
            width=600,
            height=600,
            margin=dict(l=20, r=20, t=40, b=40)
        )

        st.plotly_chart(fig_pie, use_container_width=False, width=600, height=600)

        st.subheader("📊 Expected Taxes and Costs")

        fig_bar = go.Figure(
            data=[go.Bar(
                x=["Dividend Tax", "Capital Gains Tax", "Transaction Costs"],
                y=[best_portfolio["dividend_tax"], best_portfolio["capital_gains_tax"], best_portfolio["total_transaction_costs"]],
                marker_color=["blue", "red", "green"]
            )]
        )

        fig_bar.update_layout(
            title="Expected Taxes and Costs",
            xaxis_title="Cost Type",
            yaxis_title="Total Cost (£)",
            width=800,
            height=500,
            margin=dict(l=40, r=40, t=40, b=40)
        )

        st.plotly_chart(fig_bar, use_container_width=False, width=800, height=500)



# ---- Risk Survey Page ----
elif page == "Risk Survey":
    if "is_logged_in" not in st.session_state or not st.session_state["is_logged_in"]:
        st.warning("🔒 Please log in first.")
        st.stop()

    st.title("📊 Risk Tolerance Survey")
    
    user = st.session_state["user"]
    username = user["username"]

    st.subheader("📋 Answer the Following Questions:")
    age = st.number_input("Enter your age:", min_value=18, max_value=95, value=int(user["age"]))
    kids = st.number_input("Number of kids:", min_value=0, value=int(user["kids"]))
    house = st.radio("Do you own a house?", [0, 1], index=int(user["house"]), format_func=lambda x: "Yes" if x == 1 else "No")
    married = st.radio("Are you married?", [0, 1], index=int(user["married"]), format_func=lambda x: "Yes" if x == 1 else "No")


    calculated_risk_level = risk_model(age, kids, house, married)
    
    st.subheader(f"📊 Calculated Risk Level: **{calculated_risk_level}%**")

    if st.button("Save Risk Level"):
        user["risk level"] = int(calculated_risk_level)
        save_user_data({username: user})
        st.success(f"✅ Risk level updated to {calculated_risk_level}%")
        st.rerun()
