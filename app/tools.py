from dotenv import load_dotenv
import yfinance as yf
from typing import Union, List, Dict
import requests
import json
import os

load_dotenv()
ALPHA_API_KEY = os.getenv("ALPHA_API_KEY")

def fetch_stock_info(symbol: str) -> str:
    """Fetches financial metrics like Market Cap, Revenue, and Revenue Multiple using Alpha Vantage.
    
    Args:
        symbol: the ticker symbol of the stock whose information to obtain.
    Returns:
        A string message containing the required information of the stock.
    """
    if not ALPHA_API_KEY:
        return "Error: Alpha Vantage API key is missing. Please set ALPHA_VANTAGE_API_KEY."

    try:
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
    except Exception:
        return f"Somthing went wrong. {data}"

def fetch_stock_history(ticker: str, days: int) -> Union[str, List[dict]]:
    """
    Fetch historical stock prices for a given ticker and number of days into the past.
    
    Args:
        ticker (str): Stock ticker symbol for the stock whose history is to be obtained
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
    
def currency_exchange(from_name: str, to_name: str) -> str:
    """
    Obtains the current day's currency exchange price from one to another.

    Args:
        from_name: The currency to get the exchange rate for. It can either be a physical currency or digital/crypto currency. For example: USD or BTC.
        to_name: The destination currency for the exchange rate. It can either be a physical currency or digital/crypto currency. For example: USD or BTC.
    Returns:
        A string message telling the exchange rate.
    """
    try:
        url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_name}&to_currency={to_name}&apikey={ALPHA_API_KEY}'
        r = requests.get(url)
        data = r.json()

        exchange_rate_info = data['Realtime Currency Exchange Rate']
        from_currency = exchange_rate_info['1. From_Currency Code']
        to_currency = exchange_rate_info['3. To_Currency Code']
        exchange_rate = float(exchange_rate_info['5. Exchange Rate'])  # Convert to float
        last_refreshed = exchange_rate_info['6. Last Refreshed']

        return f"Exchange Rate ({from_currency} to {to_currency}): {exchange_rate}, last refreshed on {last_refreshed}"
    except Exception:
        return f"Something went wrong. {data}"

def currency_exchange_daily(from_name: str, to_name: str, days: int) -> str:
    """
    Returns the past few days currency exchange price from one to another.

    Args:
        from_name: The currency from which to convert. Eg: EUR
        to_name: The currency into which to convert. Eg: JPY
        days: The number of days from the past to get the rates
    Returns:
        A string in json format telling the exchange rates of past few days.
    """
    try:
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
    except Exception:
        return f"Somthing went wrong. {data}"

def crypto_currency_exchange(from_name: str, to_name: str) -> str:
    """
    Obtains the current day's crypto exchange price into the country currency price.

    Args:
        from_name: The crypto currency symbol to get the exchange rate for. For example: BTC.
        to_name: The destination currency for the exchange rate. For example: USD.
    Returns:
        A string message telling the exchange rate.
    """
    try:
        url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_name}&to_currency={to_name}&apikey={ALPHA_API_KEY}'
        r = requests.get(url)
        data = r.json()

        exchange_rate_info = data['Realtime Currency Exchange Rate']
        from_currency = exchange_rate_info['1. From_Currency Code']
        to_currency = exchange_rate_info['3. To_Currency Code']
        exchange_rate = float(exchange_rate_info['5. Exchange Rate'])  # Convert to float
        last_refreshed = exchange_rate_info['6. Last Refreshed']

        return f"Exchange Rate ({from_currency} to {to_currency}): {exchange_rate}, last refreshed on {last_refreshed}"
    except Exception:
        return f"Something went wrong. {data}"

def digital_currency_daily(from_name: str, to_name: str, days: int) -> str:
    """
    Obtains the past few days exchange price of a digital/crypto currency to a physical one.

    Args:
        from_name: The digital/crypto currency whose price to obtain. Eg: BTC.
        to_name: The exchange market of of physical currency to convert to. Eg: EUR.
        days: Number of days into the past to check for the rates.
    Returns:
        A string json messages containing the exchange rates of the past few days.
    In case of error:
        Get the convertion of US dollars and use the currency_exchange_daily function to convert the obtained US dollars into that currency, use the calculator function when necessary.
    """
    try:
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
    except Exception:
        return f"Somthing went wrong. {data}"