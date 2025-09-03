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
    compute_alpha_signals,
    backtest_price_strategy,
    monte_carlo_forecast
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

@st.cache_data(show_spinner=False, ttl=1200)
def cached_full_history(t: str):
    from main import get_full_history
    return get_full_history(t)

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

tab_overview, tab_fund, tab_alpha, tab_bt, tab_fc = st.tabs(["Analysis", "Fundamentals", "Alpha Signals", "Backtest", "Forecast"])

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
                # Load full history once and slice to selected period for analysis
                full_hist = cached_full_history(ticker)
                if isinstance(full_hist, pd.DataFrame) and not full_hist.empty:
                    # Map period to pandas offset
                    period_to_days = {
                        '1mo': 31, '3mo': 93, '6mo': 186, '1y': 366, '3y': 3*366, '5y': 5*366
                    }
                    days = period_to_days.get(period, 186)
                    stock_data_for_pred = full_hist.tail(days)
                else:
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
                    from main import get_current_price, get_trend_labels, compute_trend_targets
                    last_close = float(price['Close'].iloc[-1])
                    trends = get_trend_labels(price)
                    targets = compute_trend_targets(price)
                    # Convert selected period to months label
                    def _period_to_months(p: str) -> int | None:
                        if p.endswith('mo'):
                            try:
                                return int(p[:-2])
                            except Exception:
                                return None
                        if p.endswith('y'):
                            try:
                                return int(p[:-1]) * 12
                            except Exception:
                                return None
                        return None
                    def _fmt_m(m: int | None) -> str:
                        return f"{m}M" if isinstance(m, int) else period.upper()
                    sel_months = _period_to_months(period)
                    short_label = _fmt_m(sel_months)
                    # Long-term label: a broader horizon derived from selection (4x, min 12M, max 60M)
                    long_months = None if sel_months is None else max(12, min(60, sel_months * 4))
                    long_label = _fmt_m(long_months)
                    c0, c1, c2 = st.columns(3)
                    c0.metric('Current Price', f"{last_close:.2f}")

                    def _trend_view(label: str, price_val: float, trend_label: str):
                        t = (trend_label or 'Unknown').lower()
                        if 'up' in t:
                            arrow = '▲'
                            color = '#16a34a'
                            text = 'Uptrend'
                        elif 'down' in t:
                            arrow = '▼'
                            color = '#dc2626'
                            text = 'Downtrend'
                        elif 'side' in t:
                            arrow = '↔'
                            color = '#6b7280'
                            text = 'Sideways'
                        else:
                            arrow = '↔'
                            color = '#6b7280'
                            text = 'Unknown'
                        html = f"""
                        <div class='metric-small'>
                            <div class='metric-label'>{label}</div>
                            <div class='metric-value'><span style='color:{color};margin-right:6px'>{arrow}</span> {price_val:.2f} <span class='subtle' style='margin-left:6px'>{text}</span></div>
                        </div>
                        """
                        return html

                    # Use projected targets instead of repeating current price
                    short_price_target = targets.get('short_target', last_close)
                    long_price_target = targets.get('long_target', last_close)
                    c1.markdown(_trend_view(f'Short-term Target ({short_label})', float(short_price_target), trends.get('short_term', 'Unknown')), unsafe_allow_html=True)
                    c2.markdown(_trend_view(f'Long-term Target ({long_label})', float(long_price_target), trends.get('long_term', 'Unknown')), unsafe_allow_html=True)
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

with tab_bt:
    st.subheader('Backtest (Price-only Strategy)')
    st.caption('Evaluates a simple rule: Long when Close>SMA20>SMA50 and MACD>Signal; Short when opposite. Flat otherwise. Metrics are for the selected timeframe slice.')
    data = st.session_state['store'].get(store_key, {})
    price = data.get('price')
    if isinstance(price, pd.DataFrame) and not price.empty:
        res = backtest_price_strategy(price)
        if isinstance(res, dict) and 'error' in res:
            st.error(res['error'])
        else:
            mets = res['metrics']
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric('Hit Rate', f"{mets.get('hit_rate', 0)}%")
            c2.metric('Total Return', f"{mets.get('total_return_pct', 0)}%")
            c3.metric('CAGR', f"{mets.get('cagr_pct', 0)}%")
            c4.metric('Max Drawdown', f"{mets.get('max_drawdown_pct', 0)}%")
            c5.metric('Sharpe', f"{mets.get('sharpe', 0)}")

            st.write('---')
            st.markdown('<div class="section-title">📈 Equity Curve</div>', unsafe_allow_html=True)
            eq = res['equity']
            import plotly.graph_objects as go
            figbt = go.Figure()
            figbt.add_trace(go.Scatter(x=eq.index, y=eq['Equity'], mode='lines', name='Equity'))
            figbt.update_layout(template='plotly_white', height=320, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(figbt, use_container_width=True)
    else:
        st.info('Run Analysis first to load price data for backtesting.')

with tab_fc:
    st.subheader('Forecast (Monte Carlo)')
    st.caption('Simulated future prices using GBM from historical drift/volatility. Educational, not advice.')
    data = st.session_state['store'].get(store_key, {})
    price = data.get('price')
    if isinstance(price, pd.DataFrame) and not price.empty:
        c1, c2, c3 = st.columns(3)
        horizon = c1.selectbox('Horizon', ['3M','6M','1Y'], index=2)
        sims = c2.slider('Simulations', 100, 2000, 500, step=100)
        seed = c3.number_input('Random Seed (optional)', value=0)
        with st.expander('Advanced (optional): Drift/Vol overrides'):
            colA, colB = st.columns(2)
            mu_override = colA.number_input('Annualized Drift μ (e.g., 0.10 = 10%)', value=None, step=0.01, format='%.4f')
            sigma_override = colB.number_input('Annualized Volatility σ (e.g., 0.25 = 25%)', value=None, step=0.01, format='%.4f')
        days_map = {'3M': 63, '6M': 126, '1Y': 252}
        days = days_map.get(horizon, 252)
        res = monte_carlo_forecast(
            price,
            days_ahead=days,
            sims=int(sims),
            seed=int(seed) if seed else None,
            override_mu_ann=mu_override,
            override_sigma_ann=sigma_override
        )
        if isinstance(res, dict) and 'error' in res:
            st.error(res['error'])
        else:
            pct = res['percentiles']
            import plotly.graph_objects as go
            figf = go.Figure()
            figf.add_trace(go.Scatter(x=pct.index, y=pct['p50'], mode='lines', name='Median (P50)'))
            figf.add_trace(go.Scatter(x=pct.index, y=pct['p90'], mode='lines', name='P90', line=dict(width=0), showlegend=True))
            figf.add_trace(go.Scatter(x=pct.index, y=pct['p10'], mode='lines', name='P10', fill='tonexty', line=dict(width=0)))
            figf.update_layout(template='plotly_white', height=380, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(figf, use_container_width=True)

            lp = res['params']['last_price']
            st.metric('Current Price', f"{lp:.2f}")
            st.metric('Horizon Median', f"{pct['p50'].iloc[-1]:.2f}")
            st.caption('Bands show the 10th–90th percentile range across simulations.')

            # CSV downloads
            csv_col1, csv_col2 = st.columns(2)
            csv_col1.download_button(
                label='Download Percentiles CSV',
                data=pct.to_csv().encode('utf-8'),
                file_name=f'{ticker}_{horizon}_percentiles.csv',
                mime='text/csv'
            )
            paths_df = res['paths']
            csv_col2.download_button(
                label='Download Paths CSV',
                data=paths_df.to_csv().encode('utf-8'),
                file_name=f'{ticker}_{horizon}_paths.csv',
                mime='text/csv'
            )
    else:
        st.info('Run Analysis first to load price data for forecasting.')