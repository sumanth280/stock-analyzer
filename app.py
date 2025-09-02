import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from main import get_financial_news, get_stock_data, analyze_sentiment, analyze_sentiment_detailed, predict_trend

st.title('AI News Sentiment & Stock Predictor')

st.markdown("""
This app analyzes recent financial news sentiment (FinBERT) and combines it with simple price momentum to estimate a short‑term direction. Works for U.S. and Indian stocks (try NSE symbols like RELIANCE or TCS).
""")

@st.cache_data(show_spinner=False, ttl=600)
def cached_news(t: str):
    return get_financial_news(t)

@st.cache_data(show_spinner=False, ttl=600)
def cached_prices(t: str, p: str):
    return get_stock_data(ticker=t, period=p)

# User input for the stock ticker
ticker = st.text_input('Enter a stock ticker (e.g., AAPL, RELIANCE, TCS)', 'RELIANCE').upper()
period = st.selectbox('History period', ['1mo','3mo','6mo','1y'], index=1)

if st.button('Analyze'):
    if not ticker:
        st.error("Please enter a valid stock ticker.")
    else:
        # Fetch data and analyze sentiment
        st.subheader(f'Analysing {ticker}...')
        
        # 1. Get News
        st.info('Fetching recent financial news...')
        news_headlines = cached_news(ticker)
        if isinstance(news_headlines, str):
            st.error(news_headlines)
        elif not news_headlines:
            st.warning('Could not find any recent news for this ticker.')
        else:
            news_df = pd.DataFrame(news_headlines, columns=['Headline'])
            st.write('Recent News Headlines:')
            st.dataframe(news_df, use_container_width=True)
            
            # 2. Analyze Sentiment
            st.info('Running FinBERT sentiment analysis...')
            detailed = analyze_sentiment_detailed(news_headlines)
            sentiment_results = detailed['aggregate']
            
            st.write('---')
            st.subheader('Sentiment Analysis Results')
            st.metric(label="Overall Sentiment", value=f"{sentiment_results['label']} ({sentiment_results['sentiment_score']:.2f})")
            
            # 3. Predict Trend
            st.write('---')
            st.subheader('Short-Term Trend Prediction')
            # Fetch history first for momentum-aware prediction
            stock_data_for_pred = cached_prices(ticker, period)
            if isinstance(stock_data_for_pred, str):
                predicted_trend = predict_trend(sentiment_results['sentiment_score'])
            else:
                predicted_trend = predict_trend(sentiment_results['sentiment_score'], stock_data_for_pred)
            st.success(f"Based on recent news sentiment, the predicted trend is: {predicted_trend}")

            # Per-headline table
            with st.expander('Per-headline sentiment details'):
                if isinstance(detailed, dict) and 'items' in detailed:
                    details_df = pd.DataFrame(detailed['items'])
                    if not details_df.empty:
                        details_df = details_df.rename(columns={'label': 'Label', 'score': 'Score', 'headline': 'Headline'})
                        st.dataframe(details_df[['Headline','Label','Score']], use_container_width=True, height=400)
            
            # 4. Get Stock Data and Plot
            st.info('Fetching historical stock data...')
            stock_data = cached_prices(ticker, period)
            if isinstance(stock_data, str):
                st.error(stock_data)
            else:
                st.write('---')
                st.subheader(f'Candlestick Chart ({period})')
                fig = go.Figure(data=[go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close']
                )])
                fig.update_layout(xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)