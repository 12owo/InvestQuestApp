git add requirements.txt
git commit -m "Added plotly to requirements"
git push origin main

source /home/adminuser/venv/bin/activate  # Adjust the path if needed
pip install newsapi-python

-name: Install dependencies
run: pip install -r requirements.txt

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import requests
from prophet import Prophet
import plotly.graph_objects as go
from newsapi import NewsApiClient


# Set up Streamlit app title
st.title("📈Invest Quest📉")

# API key for NewsAPI
NEWS_API_KEY = "639841a3c34342c99e0f6459447f72fa"

# Custom styling 
st.markdown(
    """
    <style>
    .Head-font {
        font-size: 30px !important;  /* Adjust the font size as needed */
        font-weight: bold;
        margin: 0;
        padding: 0;
    }

    .SubHead-font {
        font-size: 22px !important;  /* Adjust the font size as needed */
        font-weight: bold;
        margin: 0;
        padding: 0;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# List of stock tickers for selection
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "WIT", "ORCL", "IBM", "META", "TSLA"]

# Dropdown for stock selection
st.markdown("<p class='Head-font'>Select a Stock:</p>", unsafe_allow_html=True)
selected_ticker = st.selectbox("", tickers, label_visibility="collapsed")
st.write("\n")

# Fetch stock data from Yahoo Finance
#st.write(f"Fetching data for {selected_ticker}...")

data = yf.download(selected_ticker, start="2020-01-01", end="2025-01-01")

# Check if data is available
if not data.empty:
    # Reset index so that "Date" is properly formatted
    data = data.reset_index()

    # Flatten MultiIndex columns (if present)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    # Display the fetched data as a table with more rows (e.g., 20 rows)
    st.markdown(f"<p class='SubHead-font'>Stock Data for: {selected_ticker}</p>", unsafe_allow_html=True)
    st.write(data.head(20))  # Show the first 20 rows (or adjust this number)
    st.write("\n")
    st.write("\n")

    # Check the columns to ensure 'Close' exists
    #st.write(f"Columns in the data: {data.columns}")

    # Ensure "Date" is in the correct format
    data["Date"] = pd.to_datetime(data["Date"])  # Convert to datetime format if needed

    # Plot the closing price using Plotly
    st.markdown(f"<p class='SubHead-font'>{selected_ticker} Closing Price Over Time</p>", unsafe_allow_html=True)   
    fig = px.line(data, x="Date", y="Close",title="")
    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price (USD)")
    fig.update_layout(title_text=None)  # or title_text=None
    # Display the chart
    st.plotly_chart(fig)



    # News fetching for selected ticker
    st.subheader(f"Latest News for {selected_ticker}")

    # Load and display news headlines for the selected stock
    def fetch_news(ticker):
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        company = {
            "AAPL": "Apple",
            "GOOGL": "Google",
            "MSFT": "Microsoft",
            "AMZN": "Amazon",
            "NVDA": "Nvidia",
            "WIT": "Wipro",
            "ORCL": "Oracle",
            "IBM": "IBM",
            "META": "Meta",
            "TSLA": "Tesla"
        }
        query = company.get(ticker, ticker)
        news = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=10)
        return news['articles']

    def display_news(news_articles):
        st.subheader(f"Top News Headlines for {selected_ticker}")
        for i, article in enumerate(news_articles):
            st.write(f"{i + 1}. [{article['title']}]({article['url']}) - {article['source']['name']}")

    # Fetch and display news for the selected stock
    if st.button("Show News Headlines"):
        news_articles = fetch_news(selected_ticker)
        display_news(news_articles)
        st.write("\n")
        st.write("\n")
    st.write("\n")
    st.write("\n")

    # Get prediction slider (Years)
    st.markdown("<p class='SubHead-font'>Select number of years for prediction</p>", unsafe_allow_html=True)
    prediction_years = st.slider("Select number of years for prediction", 1, 10, 3, label_visibility="collapsed")
 
    # Prepare data for Prophet
    df_prophet = data[['Date', 'Close']]
    df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

    # Prepare the Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(df_prophet)

    # Make future dataframe for prediction
    future = model.make_future_dataframe(periods=365 * prediction_years, freq='D')
    forecast = model.predict(future)

    # Plot prediction
    fig_pred = model.plot(forecast)
    st.subheader(f"Stock Prediction of {selected_ticker} for {prediction_years} Years")
    st.pyplot(fig_pred)

    # Investment Input: User provides the amount they want to invest
    st.markdown(f"<p class='SubHead-font'>Enter your investment amount (in USD):</p>", unsafe_allow_html=True)
    investment_value = st.number_input("**Enter your investment amount (in USD):**", min_value=1, value=1000, label_visibility="collapsed")
    #st.markdown(<p style='font-size:20px;'> investment_value </p>, unsafe_allow_html=True)


    
    if investment_value > 0:
        # Calculate the number of shares user can buy with their investment
        initial_price = df_prophet['y'].iloc[-1]  # The last available closing price
        shares_to_buy = investment_value / initial_price  # How many shares user can buy with their investment

        # Calculate the profit at the predicted future price
        final_price = forecast['yhat'].iloc[-1]  # Predicted future price
        profit = (final_price - initial_price) * shares_to_buy  # Profit calculation

        # Show results
        st.write(f"With an investment of **${investment_value:.2f}**, you could buy **{shares_to_buy:.2f} shares**.")
        st.write(f"Predicted future price after **{prediction_years} years**: **${final_price:.2f}**")
        st.write(f"Your potential profit would be: **${profit:.2f}**")
        st.write(f"\n")
        st.write(f"\n")
    else:
        st.error("Please enter a valid investment amount.")



    # Allow multiple stock selection for comparison
    st.markdown(f"<p class='Head-font'>Compare your stocks</p>", unsafe_allow_html=True)
    selected_stocks = st.multiselect("Select stocks for comparison", tickers, default=["AAPL"], label_visibility="collapsed")
    st.write(f"\n")
    #st.write(f"\n")

    # Prediction duration slider (1-10 years)
    st.markdown('<p class="SubHead-font">Years of prediction:</p>', unsafe_allow_html=True)
    n_years = st.slider("", 1, 10, label_visibility="collapsed")
    period = n_years * 365
    st.write("\n")

    # Load stock data function (for comparison)
    @st.cache_data
    def load_data(tickers):
        data1 = yf.download(tickers, start="2020-01-01", end="2025-01-01")
        data1.reset_index(inplace=True)
        return data1

    # Prophet model forecast function
    def forecast_stock(data1, periods):
        # Check if the 'Close' column exists and is a valid Series
        #st.write(f"Columns in data for forecasting: {data.columns}")  # Debugging step


        if isinstance(data1.columns, pd.MultiIndex):
            data1.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]


        # Prepare data for Prophet
        df_train = data1[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
        
        # Fit the Prophet model
        model = Prophet(yearly_seasonality=True)
        model.fit(df_train)
        
        # Create a future dataframe for prediction
        future = model.make_future_dataframe(periods=365 * n_years, freq='D')
        # Make predictions
        forecast = model.predict(future)
        
        return forecast

    # Load and forecast for each selected stock
    def load_and_forecast_multiple_stocks(selected_stocks):
        forecasts = {}
        for stock in selected_stocks:
            data1 = load_data(stock)
            if data1 is not None:
                forecast = forecast_stock(data1, period)
                if forecast is not None:
                    forecasts[stock] = forecast
        return forecasts

    # Function to plot comparison between multiple stocks
    def plot_stock_comparison(forecasts):
        fig = go.Figure()
        for stock, forecast in forecasts.items():
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name=f"{stock} Forecast"))

        st.markdown(f"<p class='SubHead-font'>Stock Price Prediction Comparison</p>", unsafe_allow_html=True)
        fig.update_layout(title="", xaxis_rangeslider_visible=True)
        
        st.plotly_chart(fig)

    # Display comparison chart
    if selected_stocks:
        forecasts = load_and_forecast_multiple_stocks(selected_stocks)
        if forecasts:
            plot_stock_comparison(forecasts)

else:
    st.error("No data found! Please check the stock ticker symbol.")
