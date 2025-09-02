import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from main import get_financial_news, get_stock_data, analyze_sentiment, predict_trend

st.title('AI News Sentiment & Stock Predictor')

st.markdown("""
This application analyzes financial news sentiment for a given stock and provides a simple trend prediction.
""")

# User input for the stock ticker
ticker = st.text_input('Enter a stock ticker (e.g., AAPL, GOOGL, TSLA)', 'TSLA').upper()

if st.button('Analyze'):
    if not ticker:
        st.error("Please enter a valid stock ticker.")
    else:
        # Fetch data and analyze sentiment
        st.subheader(f'Analysing {ticker}...')
        
        # 1. Get News
        st.info('Scraping recent financial news...')
        news_headlines = get_financial_news(ticker)
        if isinstance(news_headlines, str):
            st.error(news_headlines)
        elif not news_headlines:
            st.warning('Could not find any recent news for this ticker.')
        else:
            news_df = pd.DataFrame(news_headlines, columns=['Headline'])
            st.write('Recent News Headlines:')
            st.dataframe(news_df, use_container_width=True)
            
            # 2. Analyze Sentiment
            st.info('Running sentiment analysis...')
            sentiment_results = analyze_sentiment(news_headlines)
            
            st.write('---')
            st.subheader('Sentiment Analysis Results')
            st.metric(label="Overall Sentiment", value=f"{sentiment_results['label']} ({sentiment_results['sentiment_score']:.2f})")
            
            # 3. Predict Trend
            st.write('---')
            st.subheader('Short-Term Trend Prediction')
            predicted_trend = predict_trend(sentiment_results['sentiment_score'])
            st.success(f"Based on recent news sentiment, the predicted trend is: {predicted_trend}")
            
            # 4. Get Stock Data and Plot
            st.info('Fetching historical stock data...')
            stock_data = get_stock_data(ticker)
            if isinstance(stock_data, str):
                st.error(stock_data)
            else:
                st.write('---')
                st.subheader('Candlestick Chart (Last Month)')
                fig = go.Figure(data=[go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close']
                )])
                fig.update_layout(xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)