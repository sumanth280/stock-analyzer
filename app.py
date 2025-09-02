import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from main import (
    get_financial_news,
    get_stock_data,
    analyze_sentiment,
    analyze_sentiment_detailed,
    predict_trend,
    advanced_predict_trend,
    get_fundamentals
)

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

@st.cache_data(show_spinner=False, ttl=600)
def cached_fundamentals(t: str):
    return get_fundamentals(t)

# User input for the stock ticker
ticker = st.text_input('Enter a stock ticker (e.g., AAPL, RELIANCE, TCS)', 'RELIANCE').upper()
period = st.selectbox('History period', ['1mo','3mo','6mo','1y'], index=1)

tab_overview, tab_fund = st.tabs(["Analysis", "Fundamentals"])

with tab_overview:
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
                    adv = {'direction': predicted_trend, 'confidence': 50, 'signals': ['Fallback: price data unavailable']}
                else:
                    predicted_trend = predict_trend(sentiment_results['sentiment_score'], stock_data_for_pred)
                    adv = advanced_predict_trend(sentiment_results['sentiment_score'], stock_data_for_pred)
                st.success(f"Predicted trend: {adv['direction']} (confidence {adv['confidence']}%)")
                with st.expander('Why this prediction?'):
                    for s in adv.get('signals', []):
                        st.write(f"- {s}")

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

with tab_fund:
    st.subheader('Fundamental Analysis')
    st.caption('Key ratios and financial statements (from Yahoo Finance).')
    if st.button('Load Fundamentals'):
        fundamentals = cached_fundamentals(ticker)
        if isinstance(fundamentals, dict) and 'error' in fundamentals:
            st.error(fundamentals['error'])
        elif isinstance(fundamentals, dict):
            ratios = fundamentals.get('ratios', {})
            cols = st.columns(3)
            pairs = list(ratios.items())
            for i, (k, v) in enumerate(pairs):
                with cols[i % 3]:
                    if isinstance(v, (int, float)) and v is not None:
                        if any(x in k for x in ['Margin', 'ROE', 'ROA']):
                            st.metric(k, f"{v*100:.1f}%")
                        elif k in ['Price']:
                            st.metric(k, f"{v:.2f}")
                        else:
                            st.metric(k, f"{v:,.0f}")
                    else:
                        st.metric(k, "—")

            st.write('---')
            ist = fundamentals.get('income_statement')
            bst = fundamentals.get('balance_sheet')
            cfs = fundamentals.get('cash_flow')

            sub1, sub2, sub3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            with sub1:
                if isinstance(ist, pd.DataFrame) and not ist.empty:
                    st.dataframe(ist, use_container_width=True)
                else:
                    st.info('No income statement data available.')
            with sub2:
                if isinstance(bst, pd.DataFrame) and not bst.empty:
                    st.dataframe(bst, use_container_width=True)
                else:
                    st.info('No balance sheet data available.')
            with sub3:
                if isinstance(cfs, pd.DataFrame) and not cfs.empty:
                    st.dataframe(cfs, use_container_width=True)
                else:
                    st.info('No cash flow data available.')