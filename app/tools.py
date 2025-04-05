import os
import sys
import json
import numpy as np
import base64
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import requests
import gradio as gr
import cloudinary
import cloudinary.uploader
import traceback
import yfinance as yf
import torch
import torch.nn as nn
import random
from sklearn.preprocessing import MinMaxScaler
from typing import Union, List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Helper function
def get_api_key():
    """Retrieve the API key stored at chatbot.py, it can be the default as well as the user entered one."""
    from app.chatbot import ALPHA_API_KEY
    return ALPHA_API_KEY

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
        
# Configure the API
genai.configure(api_key=api_key)
            
# Use the default model
model = genai.GenerativeModel('gemini-2.0-flash')

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)
# Assuming the stock prediction transformer code is in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def fetch_stock_info(symbol: str) -> str:
    """
    Fetches financial metrics like Market Cap, Revenue, and Revenue Multiple using Alpha Vantage.
    
    Args:
        symbol: The ticker symbol of US based stock.
    Returns:
        A string message containing the stock info.
    """
    ALPHA_API_KEY = get_api_key()
    if not ALPHA_API_KEY:
        return "Error: Alpha Vantage API key is missing. Please set ALPHA_API_KEY."

    # Fetch company overview
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_API_KEY}"
    response = requests.get(url)
    data = response.json()

    market_cap = int(data.get("MarketCapitalization", 0))  # Default to 0 instead of 'N/A'
    total_revenue = int(data.get("RevenueTTM", 0))
    
    revenue_multiple = market_cap / total_revenue if total_revenue else None

    # Generate insights dynamically
    insights = []

    insights.append(f"📊 **Financial Analysis of {symbol}**")
    insights.append(f"Here's an analysis of {symbol}'s financial status based on its market capitalization and total revenue:\n")

    # Market Cap
    insights.append(f"💰 **Market Capitalization:** {symbol} is valued at approximately **${market_cap:,.0f}**.")
    if market_cap > 1_000_000_000_000:
        insights.append("🔹 This places it among the largest global companies, indicating strong investor confidence.")
    elif market_cap > 100_000_000_000:
        insights.append("🔹 This is a large-cap company, showing strong stability and market influence.")
    else:
        insights.append("🔹 This is a mid or small-cap company, which may offer growth opportunities but with higher risk.")

    # Revenue
    insights.append(f"\n📈 **Total Revenue:** {symbol} generated **${total_revenue:,.0f}** in revenue.")
    if total_revenue > 500_000_000_000:
        insights.append("🔹 This is an extremely high revenue figure, showing market dominance.")
    elif total_revenue > 50_000_000_000:
        insights.append("🔹 The company has strong revenue streams, indicating stability.")
    else:
        insights.append("🔹 Revenue is moderate, and future growth will depend on market strategy.")

    # Revenue Multiple
    if revenue_multiple:
        insights.append(f"\n📊 **Revenue Multiple:** ${market_cap:,.0f} ÷ ${total_revenue:,.0f} = **{revenue_multiple:.2f}**")
        if revenue_multiple > 10:
            insights.append("🔹 Investors are pricing in significant future growth, but expectations are high.")
        elif revenue_multiple > 5:
            insights.append("🔹 The stock is fairly valued based on revenue, suggesting balanced growth.")
        else:
            insights.append("🔹 The valuation is relatively low, which may indicate an undervalued opportunity.")

    # Final Analysis
    insights.append("\n### **🚀 Key Takeaways**")
    if revenue_multiple and revenue_multiple > 10:
        insights.append("✅ Strong market confidence but with high expectations.")
    elif revenue_multiple and revenue_multiple > 5:
        insights.append("✅ Stable financials with reasonable growth potential.")
    else:
        insights.append("✅ The company may be undervalued or facing slower growth.")

    return "\n".join(insights)

def fetch_stock_history(ticker: str, days: int) -> str:
    """
    Gets the stock price of a stock few days from the past, uses Yahoo Finance.

    Args:
        ticker: The ticker symbol of the stock. Append .NS for National Stock Exchange of India, .BO for Bombay Stock Exchange of India, .L for London Stock Exchange.
        days: Number of days into the past to get the stock price.

    Returns:
        A string message containing the dates and prices.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d")

        if hist.empty:
            return f"No historical data found for {ticker}."

        hist = hist.reset_index()
        hist["Date"] = hist["Date"].dt.strftime('%Y-%m-%d')
        result = hist.to_dict(orient="records")

        formatted_data = "\n".join([
            f"📅 Date: {entry['Date']}, Close: ${round(entry['Close'], 2)}"
            for entry in result
        ])

        # 🔥 Debugging: Print final formatted response before returning
        print(f"📤 Returning Stock History:\n{formatted_data}")

        return formatted_data if formatted_data.strip() else f"No historical data found for {ticker}."

    except Exception as e:
        return f"Error fetching stock history for {ticker}: {str(e)}"
    
def get_prediction_chart(ticker: str, predictions, current_price, volatility):
    """Generate a PNG image of stock price predictions. Used by the predict price function."""
    days = np.arange(len(predictions))
    confidence_intervals = [np.sqrt(i / 252) * volatility * current_price for i in range(len(predictions))]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Confidence Intervals
    ax.fill_between(days, 
        [p[0] - 2 * confidence_intervals[i] for i, p in enumerate(predictions)], 
        [p[0] + 2 * confidence_intervals[i] for i, p in enumerate(predictions)], 
        color='lightblue', alpha=0.3, label='95% Confidence')

    ax.fill_between(days, 
        [p[0] - confidence_intervals[i] for i, p in enumerate(predictions)], 
        [p[0] + confidence_intervals[i] for i, p in enumerate(predictions)], 
        color='blue', alpha=0.3, label='68% Confidence')

    ax.plot(days, [p[0] for p in predictions], marker='o', color='darkblue', label='Predicted Price')

    ax.set_title(f"{ticker} - Price Prediction")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Predicted Price")
    ax.grid(True)
    ax.legend()

    # Convert plot to PNG in memory
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png", bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory

    img_buffer.seek(0)  # Reset buffer position to start

    # Upload image to Cloudinary
    upload_response = cloudinary.uploader.upload(img_buffer, folder="stock_predictions")

    # Get public URL
    image_url = upload_response.get("secure_url")
    
    return image_url

def predict_stock_price(ticker: str, days: int) -> str:
    """
    Predicts the stock price of the stock few days into the future. Trained on data from Yahoo Finance, and custom made transformer for prediction.

    Args:
        ticker: The ticker symbol of the stock to predict. Append .NS for National Stock Exchange of India, .BO for Bombay Stock Exchange of India, .L for London Stock Exchange.
        days: Number of days into the future to predict the price.

    Returns:
        A string message telling the stock price, also providing a link to the graph plot.
    """
    try:
        print(days)
        from app.stock_prediction_transformer import StockManager
        stock_manager = StockManager()
        
        # Ensure models directory exists
        models_dir = ".\\models"
        if not os.path.exists(models_dir):
            try:
                os.makedirs(models_dir)
                print(f"📁 Created models directory: {models_dir}")
            except Exception as dir_error:
                print(f"❌ Error creating models directory: {dir_error}")
                return f"❌ Failed to create models directory: {dir_error}"

        model_path = os.path.join(models_dir, f"{ticker}_model.pt")
        print(f"🔍 Checking model path: {model_path}")

        if not os.path.exists(model_path):
            print(f"⚠️ No pre-trained model found for {ticker}. Training now...")
            try:
                print("before adding stock")
                add_msg = stock_manager.add_stock(ticker)
                print(f"📌 Add Stock Result: {add_msg}")

                if "Failed" in add_msg:
                    return f"❌ Failed to add stock {ticker}. Try again."

                train_msg = stock_manager.train_stock(ticker)
                print(f"📌 Train Result: {train_msg}")

                eval_msg, _ = stock_manager.evaluate_stock(ticker)
                print(f"📌 Evaluation Result: {eval_msg}")
            except Exception as train_error:
                print(f"❌ Training error: {train_error}")
                return f"❌ Error during model training: {train_error}"

        # Pass the correctly formatted duration instead of `days`
        try:
            _, predictions, info = stock_manager.predict_stock(ticker, days)
        except Exception as pred_error:
            print(f"❌ Prediction error: {pred_error}")
            return f"❌ Error during stock prediction: {pred_error}"

        print(f"📊 Raw Predictions for {days}: {predictions}")

        if predictions is None or len(predictions) == 0:
            return "❌ Model did not generate valid multi-day predictions."

        predicted_price = round(predictions[-1][0], 2)
        current_price = info.get("current_price", 0)
        accuracy = round(info.get("accuracy", 0) * 100, 2)
        volatility = info.get("volatility", 0.3)
        
        img_url = get_prediction_chart(ticker, predictions, current_price, volatility)
        
        if current_price > 0:
            percent_change = round(((predicted_price - current_price) / current_price) * 100, 2)
            change_str = f"{percent_change:+.2f}% {'increase' if percent_change > 0 else 'decrease'}"
        else:
            change_str = "change (Current price unavailable)"

        # Conversational response with line breaks and emojis
        response = (
            f"🚀 **Prediction:** The price of **{ticker}** is expected to be **${predicted_price:.2f}** in {readable_duration}.\n\n"
            f"💰 **Current Price:** ${current_price:.2f}, so we anticipate it to go {change_str}.\n\n"
            f"📊 **Model Accuracy:** {accuracy:.2f}%. While I try to be precise, the market can be unpredictable! 🎢\n\n"
            f"📈 **Prediction Chart:** [View Graph]({img_url})"
        )
        
        return str(response)

    except Exception as e:
        import traceback
        print("🔥 Error during prediction:", traceback.format_exc())
        return f"❌ Error predicting stock price for {ticker}: {str(e)}"

def currency_exchange(from_name: str, to_name: str) -> str:
    """
    Obtains the current day's currency exchange price from one to another. Obtains data from Alpha Vantage.

    Args:
        from_name: The currency to get the exchange rate for. It can either be a physical currency or digital/crypto currency. For example: USD or BTC.
        to_name: The destination currency for the exchange rate. It can either be a physical currency or digital/crypto currency. For example: USD or BTC.
    Returns:
        A string message telling the exchange rate.
    """
    ALPHA_API_KEY = get_api_key()
    if not ALPHA_API_KEY:
        return "Error: Alpha Vantage API key is missing. Please set ALPHA_API_KEY."
    
    url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_name}&to_currency={to_name}&apikey={ALPHA_API_KEY}'
    r = requests.get(url)
    data = r.json()

    exchange_rate_info = data['Realtime Currency Exchange Rate']
    from_currency = exchange_rate_info['1. From_Currency Code']
    to_currency = exchange_rate_info['3. To_Currency Code']
    exchange_rate = float(exchange_rate_info['5. Exchange Rate'])  # Convert to float
    last_refreshed = exchange_rate_info['6. Last Refreshed']

    return f"Exchange Rate ({from_currency} to {to_currency}): {exchange_rate}, last refreshed on {last_refreshed}"

def currency_exchange_daily(from_name: str, to_name: str, days: int) -> str:
    """
    Returns the past few days currency exchange price from one to another. Obtains data from Alpha Vantage.

    Args:
        from_name: The currency from which to convert. Eg: EUR
        to_name: The currency into which to convert. Eg: JPY
        days: The number of days from the past to get the rates
    Returns:
        A string in json format telling the exchange rates of past few days.
    """
    ALPHA_API_KEY = get_api_key()
    if not ALPHA_API_KEY:
        return "Error: Alpha Vantage API key is missing. Please set ALPHA_API_KEY."
    
    url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_name}&to_symbol={to_name}&apikey={ALPHA_API_KEY}'
    r = requests.get(url)
    data = r.json()

    time_series = data["Time Series FX (Daily)"]

    # Convert dates to a sorted list
    dates = sorted(time_series.keys(), reverse=True)

    # Get the last 7 available days
    last_7_days = dates[:7]

    # Format output
    formatted_data = {date: time_series[date] for date in last_7_days}

    # Print formatted data
    return json.dumps(formatted_data, indent=4)

def crypto_currency_exchange(from_name: str, to_name: str) -> str:
    """
    Obtains the current day's crypto exchange price into the country currency price. Obtains data from Alpha Vantage.

    Args:
        from_name: The crypto currency symbol to get the exchange rate for. For example: BTC.
        to_name: The destination currency for the exchange rate. For example: USD.
    Returns:
        A string message telling the exchange rate.
    """
    ALPHA_API_KEY = get_api_key()
    if not ALPHA_API_KEY:
        return "Error: Alpha Vantage API key is missing. Please set ALPHA_API_KEY."
    
    url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_name}&to_currency={to_name}&apikey={ALPHA_API_KEY}'
    r = requests.get(url)
    data = r.json()

    exchange_rate_info = data['Realtime Currency Exchange Rate']
    from_currency = exchange_rate_info['1. From_Currency Code']
    to_currency = exchange_rate_info['3. To_Currency Code']
    exchange_rate = float(exchange_rate_info['5. Exchange Rate'])  # Convert to float
    last_refreshed = exchange_rate_info['6. Last Refreshed']

    return f"Exchange Rate ({from_currency} to {to_currency}): {exchange_rate}, last refreshed on {last_refreshed}"

def digital_currency_daily(from_name: str, to_name: str, days: int) -> str:
    """
    Obtains the past few days exchange price of a digital/crypto currency to a physical one. Obtains data from Alpha Vantage.

    Args:
        from_name: The digital/crypto currency whose price to obtain. Eg: BTC.
        to_name: The exchange market of of physical currency to convert to. Eg: EUR.
        days: Number of days into the past to check for the rates.
    Returns:
        A string json messages containing the exchange rates of the past few days.
    In case of error:
        Get the convertion of US dollars and use the currency_exchange_daily function to convert the obtained US dollars into that currency, use the calculator function when necessary.
    """
    ALPHA_API_KEY = get_api_key()
    if not ALPHA_API_KEY:
        return "Error: Alpha Vantage API key is missing. Please set ALPHA_API_KEY."

    url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={from_name}&market={to_name}&apikey={ALPHA_API_KEY}'
    r = requests.get(url)
    data = r.json()

    time_series = data["Time Series (Digital Currency Daily)"]

    # Sort dates in descending order (most recent first)
    dates = sorted(time_series.keys(), reverse=True)

    # Get the last 7 available days
    last_7_days = dates[:7]

    # Format output
    formatted_data = {date: time_series[date] for date in last_7_days}

    # Print formatted data
    return json.dumps(formatted_data, indent=4)

def calculation(num1: float, num2:float, mode: str) -> Union[str, float]:
    """
    Used for basic mathemathics calculation.

    Args:
        num1: The first number in the equation.
        num2: The second number in the equation.
        mode: What type of operation to do. Available are Addition, Subtraction, Multiplication, Division, Floor division and Remainder.
    mode:
        addition: returns num1 + num2
        subtraction: returns num1 - num2
        multiplication: returns num1 * num2
        division: returns num1 / num2
        floor: for floor division, returns num1 // num2
        remainder: returns num1 % num2
    Returns:
        A float carring the result.
    In case of error:
        Will return a string message
    """
    if mode.lower() == "addition":
        return num1 + num2
    elif mode.lower() == "subtraction":
        return num1 - num2
    elif mode.lower() == "multiplication":
        return num1 * num2
    elif mode.lower() == "division":
        if num2 == 0:
            return "Division by zero error."
        return num1 / num2
    elif mode.lower() == "floor":
        return num1 // num2
    elif mode.lower() == "remainder":
        return num1 % num2
    else:
        return "Invalid symbol, or might not be implemented yet." 

def loan_calculator(principal: float, annual_rate: float, years: int) -> Dict[str, float]:
    """
    Calculate monthly loan payment, total payment, and total interest for a general loan like personal loan, auto loan and student loan, but strictly excluding home mortgage. Uses loan amortization formula to calculate the monthly payment.
    
    Args:
        principal: Total loan amount (P)
        annual_rate: Annual interest rate in percentage (r)
        years: Loan term in years (n)
    Return: 
        A dictionary with payment details.
    """
    # Convert annual interest rate to monthly and decimal format
    monthly_rate = (annual_rate / 100) / 12
    total_payments = years * 12

    # Calculate Monthly Payment using the formula
    if monthly_rate > 0:
        monthly_payment = (principal * monthly_rate * (1 + monthly_rate) ** total_payments) / \
                          ((1 + monthly_rate) ** total_payments - 1)
    else:
        monthly_payment = principal / total_payments  # If interest rate is 0

    # Calculate total cost and total interest
    total_cost = monthly_payment * total_payments
    total_interest = total_cost - principal

    return {
        "Monthly Payment": round(monthly_payment, 2),
        "Total Cost": round(total_cost, 2),
        "Total Interest Paid": round(total_interest, 2)
    }

def mortgage_calculator(principal: float, annual_rate: float, years: int) -> Dict[str, float | List[Dict[str, float]]]:
    """
    Calculate monthly mortgage payment, total payment, total interest, and provide an amortization schedule only for home mortgage. Uses the amortization formula for calculation.

    Args:
        principal: Total loan amount (P)
        annual_rate: Annual interest rate in percentage (r)
        years: Loan term in years (n)
    Returns: 
        A dictionary with mortgage details and amortization schedule.
    """
    # Convert annual interest rate to monthly and decimal format
    monthly_rate = (annual_rate / 100) / 12
    total_payments = years * 12  # Total number of monthly payments

    # Calculate Monthly Payment
    if monthly_rate > 0:
        monthly_payment = (principal * monthly_rate * (1 + monthly_rate) ** total_payments) / \
                          ((1 + monthly_rate) ** total_payments - 1)
    else:
        monthly_payment = principal / total_payments  # For 0% interest rate

    # Initialize amortization schedule
    balance = principal
    amortization_schedule = []

    # Generate Amortization Schedule
    for month in range(1, total_payments + 1):
        interest_payment = balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        balance -= principal_payment

        # Store monthly breakdown
        amortization_schedule.append({
            "Month": month,
            "Interest Paid": round(interest_payment, 2),
            "Principal Paid": round(principal_payment, 2),
            "Remaining Balance": round(balance, 2)
        })

    # Calculate total amount paid and total interest
    total_paid = monthly_payment * total_payments
    total_interest = total_paid - principal

    return {
        "Monthly Payment": round(monthly_payment, 2),
        "Total Paid": round(total_paid, 2),
        "Total Interest Paid": round(total_interest, 2),
        "Amortization Schedule": amortization_schedule  # Full breakdown per month
    }

def top_gainers_losers() -> str:
    """
    Returns the top 10 gainers and losers information in the stock market for the current date. Obtains data from Alpha Vantage.
    """
    ALPHA_API_KEY = get_api_key()
    if not ALPHA_API_KEY:
        return "Error: Alpha Vantage API key is missing. Please set ALPHA_API_KEY."
    
    try:
        url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={ALPHA_API_KEY}'
        r = requests.get(url)
        data = r.json()
        top_gainers = [
            {
                "Ticker": stock["ticker"],
                "Price": stock["price"],
                "Change %": stock["change_percentage"],
                "Volume": stock["volume"],
            }
            for stock in data.get("top_gainers", [])
        ]
        top_losers = [
            {
                "Ticker": stock["ticker"],
                "Price": stock["price"],
                "Change %": stock["change_percentage"],
                "Volume": stock["volume"],
            }
            for stock in data.get("top_losers", [])
        ]
        most_traded = [
            {
                "Ticker": stock["ticker"],
                "Price": stock["price"],
                "Volume": stock["volume"],
            }
            for stock in data.get("most_actively_traded", [])
        ]
        extracted_data = {
            "Last Updated": data.get("last_updated", "N/A"),
            "Top Gainers": top_gainers[:10],  # Top 10 for brevity
            "Top Losers": top_losers[:10],    # Top 10 for brevity
            "Most Actively Traded": most_traded[:10],  # Top 10
        }

        return str(extracted_data)
    except Exception:
        return f"Something went wrong. {data}"

def fetch_commodity_history(ticker: str, days: int) -> Union[str, List[dict]]:
    """
    Fetch historical commodity prices for a given ticker and number of days into the past. Obtains data from Yahoo Finance.
    
    Args:
        ticker (str): Commodity ticker symbol for the stock whose history is to be obtained
            Gold GC=F
            Silver SI=F
            Platinum PL=F
            Copper: HG=F
        days (int): Number of days of historical data to fetch
    Returns:
        json serializeable dictionary containing columns Date, open, high, Low, Close, Volume, Dividends, Stock Splits.
    """
    try:
        # Use yfinance to download stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period = f"{days}d")
        
        return hist.reset_index().to_dict(orient="records")
    
    except Exception as e:
        return f"Error fetching stock history for {ticker}: {str(e)}"
    
def get_current_datetime() -> str:
    """
    Returns current date and time of the local machine.
    """
    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    
    return f"CUrrent Date and Time: {formatted}"

# Example usage
if __name__ == "__main__":
    # Demonstration of the functions
    test_ticker = "AAPL"  # Example ticker
    
    # Fetch historical prices
    history = fetch_stock_history(test_ticker, 30)
    print(f"Historical prices for {test_ticker}: {history}")
    
    # Predict future prices
    predictions = predict_stock_price(test_ticker, 5)
    print(f"Predicted prices for {test_ticker}: {predictions}")
