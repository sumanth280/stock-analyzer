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
    get_fundamentals,
    get_financial_news_meta,
    compute_alpha_signals
)

st.set_page_config(page_title='AI Stock Analyzer', page_icon='📈', layout='wide')

# Minimal custom styling for a cleaner look
st.markdown(
    """
    <style>
    .main .block-container{padding-top:1.5rem;padding-bottom:2rem;}
    .stTabs [data-baseweb="tab"]{height:48px; padding: 0 18px;}
    .stTabs [data-baseweb="tab"] span{font-weight:600;}
    .metric-small .metric-value{font-size:20px!important}
    .metric-small .metric-label{font-size:12px!important;color:#6b7280!important}
    .section-title{font-weight:700; font-size:1.2rem; margin-top:0.25rem}
    .subtle{color:#6b7280}
    .divider{height:1px;background:linear-gradient(to right,transparent,#e5e7eb,transparent);margin:12px 0}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('📊 AI News Sentiment & Stock Predictor')

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

@st.cache_data(show_spinner=False, ttl=600)
def cached_news_meta(t: str):
    return get_financial_news_meta(t)

# User inputs (moved to sidebar for a cleaner main area)
with st.sidebar:
    st.header('⚙️ Controls')
    ticker = st.text_input('Enter stock', '', placeholder='e.g., AAPL, RELIANCE, TCS').upper()
    period = st.selectbox('History period', ['1mo','3mo','6mo','1y','3y','5y'], index=1)
    st.caption('Tip: Indian symbols auto-resolve to NSE/BSE')

# In-memory store so users can switch tabs without re-clicking buttons
if 'store' not in st.session_state:
    st.session_state['store'] = {}
store_key = f"{ticker}|{period}"

tab_overview, tab_fund, tab_alpha = st.tabs(["Analysis", "Fundamentals", "Alpha Signals"])

with tab_overview:
    analyze_clicked = st.button('Analyze')
    if not ticker:
        st.error("Please enter a valid stock ticker.")
    else:
        # If Analyze clicked or no stored data for this selection, compute and store
        if analyze_clicked or store_key not in st.session_state['store']:
            st.subheader(f'Analysing {ticker}...')
            st.info('Fetching recent financial news...')
            news_headlines = cached_news(ticker)
            if not isinstance(news_headlines, list) or len(news_headlines) == 0:
                st.session_state['store'][store_key] = {'error': 'No news available.'}
            else:
                detailed = analyze_sentiment_detailed(news_headlines)
                sentiment_results = detailed['aggregate']
                stock_data_for_pred = cached_prices(ticker, period)
                if isinstance(stock_data_for_pred, str):
                    predicted_trend = predict_trend(sentiment_results['sentiment_score'])
                    adv = {'direction': predicted_trend, 'confidence': 50, 'signals': ['Fallback: price data unavailable']}
                    price_for_chart = None
                else:
                    predicted_trend = predict_trend(sentiment_results['sentiment_score'], stock_data_for_pred)
                    adv = advanced_predict_trend(sentiment_results['sentiment_score'], stock_data_for_pred)
                    price_for_chart = stock_data_for_pred
                st.session_state['store'][store_key] = {
                    'news': news_headlines,
                    'detailed': detailed,
                    'sentiment': sentiment_results,
                    'adv': adv,
                    'price': price_for_chart
                }

        data = st.session_state['store'].get(store_key, {})
        if 'error' in data:
            st.warning(data['error'])
        elif data:
            # Price and trends at top
            price = data.get('price')
            if isinstance(price, pd.DataFrame) and not price.empty:
                try:
                    from main import get_current_price, get_trend_labels
                    last_close = float(price['Close'].iloc[-1])
                    trends = get_trend_labels(price)
                    c0, c1, c2 = st.columns(3)
                    c0.metric('Current Price', f"{last_close:.2f}")
                    c1.metric('Short-term Trend', f"{trends.get('short_term', 'Unknown')} @ {last_close:.2f}")
                    c2.metric('Long-term Trend', f"{trends.get('long_term', 'Unknown')} @ {last_close:.2f}")
                except Exception:
                    pass
            else:
                # Fallback: try to display current price even if history missing
                try:
                    from main import get_current_price
                    cp = get_current_price(ticker)
                    if cp is not None:
                        c0, _, _ = st.columns(3)
                        c0.metric('Current Price', f"{cp:.2f}")
                except Exception:
                    pass

            st.markdown('<div class="section-title">🧠 Sentiment Analysis</div>', unsafe_allow_html=True)
            sres = data['sentiment']
            st.metric(label="Overall Sentiment", value=f"{sres['label']} ({sres['sentiment_score']:.2f})")

            st.write('---')
            st.markdown('<div class="section-title">🔮 Short-Term Trend</div>', unsafe_allow_html=True)
            adv = data['adv']
            st.success(f"Predicted trend: {adv['direction']} (confidence {adv['confidence']}%)")
            with st.expander('Why this prediction?'):
                for s in adv.get('signals', []):
                    st.write(f"- {s}")

            st.info('Fetching historical stock data...')
            stock_data = data.get('price')
            if isinstance(stock_data, pd.DataFrame) and not stock_data.empty:
                st.write('---')
                st.markdown(f'<div class="section-title">🕯️ Candlestick Chart ({period})</div>', unsafe_allow_html=True)
                fig = go.Figure(data=[go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close']
                )])
                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    template='plotly_white',
                    height=520,
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
                # Current price and uptrend trigger/target
                try:
                    from main import get_current_price, compute_uptrend_levels
                    levels = compute_uptrend_levels(stock_data)
                    cp = levels.get('last_close', None)
                    trigger = levels.get('trigger_price', None)
                    target = levels.get('near_term_target', None)
                    if cp and trigger and target:
                        c1, c2, c3 = st.columns(3)
                        c1.metric('Current Price', f"{cp:.2f}")
                        c2.metric('Uptrend Trigger', f"{trigger:.2f}")
                        c3.metric('Near-term Target', f"{target:.2f}")
                        st.caption('Trigger = breakout above recent 20‑day high with small buffer. Target ≈ last close + ATR.')
                except Exception:
                    pass
            else:
                st.info('Price data not available for chart.')
                # Show current price even when we cannot plot the chart
                try:
                    from main import get_current_price
                    cp = get_current_price(ticker)
                    if cp is not None:
                        c1, _, _ = st.columns(3)
                        c1.metric('Current Price', f"{cp:.2f}")
                except Exception:
                    pass

            st.write('---')
            news_df = pd.DataFrame(data['news'], columns=['Headline'])
            st.write('Recent News Headlines:')
            st.dataframe(news_df, use_container_width=True)

            with st.expander('Per-headline sentiment details'):
                detailed = data['detailed']
                if isinstance(detailed, dict) and 'items' in detailed:
                    details_df = pd.DataFrame(detailed['items'])
                    if not details_df.empty:
                        details_df = details_df.rename(columns={'label': 'Label', 'score': 'Score', 'headline': 'Headline'})
                        st.dataframe(details_df[['Headline','Label','Score']], use_container_width=True, height=400)

with tab_fund:
    st.markdown('<div class="section-title">📚 Fundamental Analysis</div>', unsafe_allow_html=True)
    st.caption('Key ratios and financial statements (from Yahoo Finance).')
    # Load or reuse fundamentals in store
    data = st.session_state['store'].get(store_key, {})
    if 'fundamentals' not in data:
        fundamentals = cached_fundamentals(ticker)
        if store_key not in st.session_state['store']:
            st.session_state['store'][store_key] = {}
        st.session_state['store'][store_key]['fundamentals'] = fundamentals
        data = st.session_state['store'][store_key]
    fundamentals = data.get('fundamentals', {})
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
            st.write('---')
            with st.expander('What do these fundamentals mean?'):
                st.markdown('- **P/E (Price to Earnings)**: How much investors pay for ₹1 of earnings. Lower can be cheaper, but growth matters.')
                st.markdown('- **P/S (Price to Sales)**: Price versus revenue. Useful for low-profit or early-growth firms.')
                st.markdown('- **P/B (Price to Book)**: Price versus net assets. Below 1 can indicate value (with caveats).')
                st.markdown('- **ROE (Return on Equity)**: Profit versus shareholder equity. Higher often indicates efficient business.')
                st.markdown('- **ROA (Return on Assets)**: Profit versus total assets. Compares profitability across capital intensity.')
                st.markdown('- **Debt/Equity**: Leverage level. Higher = more debt risk; sector norms vary.')
                st.markdown('- **Current Ratio**: Short‑term liquidity (Current Assets / Current Liabilities). Below 1 can be tight.')
                st.markdown('- **EBIT/Net/FCF Margins**: Profitability relative to sales at different stages (operating, bottom line, cash flow).')

with tab_alpha:
    st.subheader('Alpha Signals (Unique Feature)')
    st.caption('Opportunity score, sentiment-price divergence, risk (ATR%), and recent news heat.')
    # Use or compute alpha signals without button
    data = st.session_state['store'].get(store_key, {})
    # Ensure price and sentiment available
    if 'price' not in data or data.get('price') is None:
        price = cached_prices(ticker, period)
        if store_key not in st.session_state['store']:
            st.session_state['store'][store_key] = {}
        if not isinstance(price, str):
            st.session_state['store'][store_key]['price'] = price
        data = st.session_state['store'][store_key]
    if 'sentiment' not in data:
        headlines = cached_news(ticker)
        detailed_alpha = analyze_sentiment_detailed(headlines) if headlines else {'aggregate': {'sentiment_score': 0.0}}
        st.session_state['store'][store_key]['sentiment'] = detailed_alpha['aggregate']
        st.session_state['store'][store_key]['detailed'] = detailed_alpha
        data = st.session_state['store'][store_key]
    # Compute alpha if not present
    if 'alpha' not in data:
        meta_news = cached_news_meta(ticker)
        alpha = compute_alpha_signals(data.get('price'), data.get('sentiment', {}), meta_news)
        st.session_state['store'][store_key]['alpha'] = alpha
        data = st.session_state['store'][store_key]

    alpha = data.get('alpha', {})
    if isinstance(alpha, dict) and 'error' in alpha:
        st.error(alpha['error'])
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric('Opportunity Score', alpha.get('opportunity_score', 0))
            st.caption('ℹ️ Higher = better short-term setup')
        with c2:
            st.metric('Divergence (Sentiment vs 5d Return)', alpha.get('divergence_score', 0))
            st.caption('ℹ️ Positive: news > price; Negative: price > news')
        with c3:
            st.metric('Risk Index (ATR%)', alpha.get('risk_index', 0))
            st.caption('ℹ️ Higher = more volatile/jumpy')
        with c4:
            st.metric('News Heat (48h)', alpha.get('event_heat_48h', 0))
            st.caption('ℹ️ More headlines = more attention/events')
        st.caption('Higher Opportunity indicates stronger confluence of signals. High Risk suggests large typical daily ranges. Divergence flags a mismatch between news and price action.')
        st.write('---')
        with st.expander('What do these mean?'):
            st.markdown('- **Opportunity Score**: 0–100. Higher number = better short-term setup. It mixes: recent news mood, if price looks stretched, and if a trend change might be near.')
            st.markdown('- **Divergence**: News vs price. Positive = news is good but price hasn’t moved much yet (could catch up). Negative = price ran up but news isn’t strong (could cool off).')
            st.markdown('- **Risk Index (ATR%)**: How jumpy the stock is. Higher = bigger daily swings → more risk. Lower = calmer moves.')
            st.markdown('- **News Heat (48h)**: How much the stock is in the news recently. Bigger number = more attention/events → faster moves possible.')