import pandas as pd
import numpy as np
import yfinance as yf
from transformers import pipeline
from functools import lru_cache
import requests
import feedparser

@lru_cache(maxsize=1)
def _get_sentiment_pipeline():
    """Load FinBERT tone model once and cache the pipeline."""
    # FinBERT tone provides POSITIVE/NEGATIVE/NEUTRAL on financial text
    return pipeline(
        task='sentiment-analysis',
        model='yiyanghkust/finbert-tone',
        tokenizer='yiyanghkust/finbert-tone'
    )

def _resolve_ticker_symbol(user_symbol: str) -> str:
    """Try raw, NSE (.NS), and BSE (.BO) variants to find a ticker with data/news."""
    candidates = [user_symbol.upper(), f"{user_symbol.upper()}.NS", f"{user_symbol.upper()}.BO"]
    for symbol in candidates:
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="5d")
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                return symbol
        except Exception:
            pass
    # fallback to raw input
    return user_symbol.upper()


def get_stock_data(ticker: str, period: str = "3mo"):
    """Fetch historical OHLCV for a given ticker (auto-resolving NSE/BSE suffixes)."""
    try:
        resolved = _resolve_ticker_symbol(ticker)
        stock = yf.Ticker(resolved)
        hist = stock.history(period=period, interval="1d")
        if not isinstance(hist, pd.DataFrame) or hist.empty:
            return f"No price data found for {resolved}."
        return hist
    except Exception as e:
        return f"Error fetching stock data: {e}"


def get_full_history(ticker: str) -> pd.DataFrame | str:
    """Fetch maximum available daily history once for a ticker."""
    try:
        resolved = _resolve_ticker_symbol(ticker)
        stock = yf.Ticker(resolved)
        hist = stock.history(period="max", interval="1d")
        if not isinstance(hist, pd.DataFrame) or hist.empty:
            return f"No price data found for {resolved}."
        return hist
    except Exception as e:
        return f"Error fetching full history: {e}"

def get_financial_news(ticker: str):
    """Fetch recent news titles. Try yfinance news; fallback to Google News RSS."""
    try:
        resolved = _resolve_ticker_symbol(ticker)
        stock = yf.Ticker(resolved)
        news_items = getattr(stock, 'news', None)
        titles: list[str] = []
        if news_items:
            for item in news_items:
                title = item.get('title')
                if title:
                    titles.append(title.strip())
        # Fallback to Google News RSS if not enough headlines
        if len(titles) < 5:
            query = f"{ticker} stock OR shares OR results"
            url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if getattr(entry, 'title', None):
                    titles.append(entry.title.strip())
        # Deduplicate while preserving order
        seen = set()
        unique_titles = []
        for t in titles:
            if t not in seen:
                seen.add(t)
                unique_titles.append(t)
        return unique_titles[:30]
    except Exception as e:
        return f"Error fetching news: {e}"

def analyze_sentiment(text_list):
    """Analyze sentiment of headlines using FinBERT. Returns aggregate score and label.

    Score: sum(positive) - sum(negative). Neutral contributes 0.
    """
    if not text_list:
        return {'sentiment_score': 0.0, 'label': 'Neutral'}

    nlp = _get_sentiment_pipeline()
    results = nlp(list(text_list))

    positive_score = sum(r['score'] for r in results if r['label'].upper() == 'POSITIVE')
    negative_score = sum(r['score'] for r in results if r['label'].upper() == 'NEGATIVE')
    sentiment_score = float(positive_score - negative_score)

    if sentiment_score > 1.0:
        final_label = 'Strongly Positive'
    elif sentiment_score > 0.2:
        final_label = 'Positive'
    elif sentiment_score < -1.0:
        final_label = 'Strongly Negative'
    elif sentiment_score < -0.2:
        final_label = 'Negative'
    else:
        final_label = 'Neutral'
        
    return {'sentiment_score': sentiment_score, 'label': final_label}


def analyze_sentiment_detailed(text_list: list[str]):
    """Return per-headline sentiment results and overall aggregate."""
    if not text_list:
        return {
            'items': [],
            'aggregate': {'sentiment_score': 0.0, 'label': 'Neutral'}
        }
    nlp = _get_sentiment_pipeline()
    results = nlp(list(text_list))

    detailed = []
    positive_score = 0.0
    negative_score = 0.0
    for headline, r in zip(text_list, results):
        label = r['label'].upper()
        score = float(r['score'])
        detailed.append({'headline': headline, 'label': label, 'score': score})
        if label == 'POSITIVE':
            positive_score += score
        elif label == 'NEGATIVE':
            negative_score += score
    sentiment_score = float(positive_score - negative_score)
    if sentiment_score > 1.0:
        final_label = 'Strongly Positive'
    elif sentiment_score > 0.2:
        final_label = 'Positive'
    elif sentiment_score < -1.0:
        final_label = 'Strongly Negative'
    elif sentiment_score < -0.2:
        final_label = 'Negative'
    else:
        final_label = 'Neutral'
    return {
        'items': detailed,
        'aggregate': {'sentiment_score': sentiment_score, 'label': final_label}
    }


def get_financial_news_meta(ticker: str):
    """Like get_financial_news, but include timestamps when available.

    Returns list of dicts: { 'title': str, 'published': datetime or None }
    """
    from datetime import datetime, timezone
    out: list[dict] = []
    try:
        resolved = _resolve_ticker_symbol(ticker)
        stock = yf.Ticker(resolved)
        news_items = getattr(stock, 'news', None)
        if news_items:
            for item in news_items:
                title = item.get('title')
                ts = item.get('providerPublishTime') or item.get('providerPublishTimeMs')
                published = None
                try:
                    if ts:
                        # yfinance returns seconds epoch
                        published = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                except Exception:
                    published = None
                if title:
                    out.append({'title': title.strip(), 'published': published})
        if len(out) < 5:
            # fallback to Google RSS
            query = f"{ticker} stock OR shares OR results"
            url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = getattr(entry, 'title', None)
                pub = None
                try:
                    if getattr(entry, 'published_parsed', None):
                        pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    pub = None
                if title:
                    out.append({'title': title.strip(), 'published': pub})
        # dedupe by title
        seen = set()
        uniq = []
        for it in out:
            t = it.get('title')
            if t and t not in seen:
                seen.add(t)
                uniq.append(it)
        return uniq[:40]
    except Exception as e:
        return []


def compute_alpha_signals(price_history: pd.DataFrame, sentiment_aggregate: dict, news_items: list[dict]) -> dict:
    """Compute unique alpha-style signals: opportunity score, divergence, and risk.

    - Divergence: sentiment vs 5d price return
    - Opportunity: blend of sentiment strength, RSI extreme proximity, MACD crossover proximity
    - Risk: ATR% of price
    - Event heat: number of headlines last 48h
    """
    if not isinstance(price_history, pd.DataFrame) or price_history.empty:
        return {'error': 'No price history'}

    df = _compute_indicators(price_history)
    last = df.iloc[-1]
    close = float(last['Close']) if pd.notna(last.get('Close')) else None
    atrp = None
    if close and pd.notna(last.get('ATR14')) and last['ATR14']:
        atrp = float(last['ATR14']) / close

    # 5d return
    five_back = df['Close'].shift(5).iloc[-1] if len(df) >= 6 else None
    ret5 = None
    if close and five_back and five_back != 0:
        ret5 = (close / float(five_back)) - 1.0

    sent_score = float(sentiment_aggregate.get('sentiment_score', 0.0) or 0.0)

    # Divergence: positive when sentiment strong but price weak (and vice versa)
    if ret5 is None:
        divergence = 0.0
    else:
        divergence = float(sent_score) - float(ret5)
    divergence_score = max(-1.0, min(1.0, divergence)) * 100.0

    # Opportunity: components 0..100
    comp = 0.0
    weight = 0.0

    # Sentiment strength component
    comp += min(1.0, abs(sent_score) / 1.5) * 35.0
    weight += 35.0

    # RSI proximity to extremes: closer to 30 or 70 increases opportunity
    rsi = float(last['RSI14']) if pd.notna(last.get('RSI14')) else None
    if rsi is not None:
        prox = max(0.0, (70 - rsi) / 40.0) if rsi >= 50 else max(0.0, (rsi - 30) / 40.0)
        comp += prox * 25.0
        weight += 25.0

    # MACD cross proximity: |MACD - Signal| small -> higher opportunity (potential cross)
    if pd.notna(last.get('MACD')) and pd.notna(last.get('MACD_SIGNAL')):
        gap = abs(float(last['MACD']) - float(last['MACD_SIGNAL']))
        # normalize by recent MACD range
        macd_series = df['MACD'].dropna()
        if len(macd_series) > 10:
            rng = float(macd_series.tail(60).max() - macd_series.tail(60).min()) or 1.0
        else:
            rng = 1.0
        inv_gap = max(0.0, 1.0 - min(1.0, gap / max(1e-6, rng)))
        comp += inv_gap * 25.0
        weight += 25.0

    # Valuation tilt: if P/E or P/S is below median of last 4 periods -> mild bump (approx)
    # We do not have rolling valuation; skip or set neutral
    comp += 0.0
    weight += 15.0  # leave some headroom to keep max at 100

    opportunity_score = int(min(100.0, comp))

    # Risk index from ATR%
    risk_index = int(min(100.0, (atrp or 0.0) / 0.06 * 100.0))  # 6% ATR% ~ 100 risk

    # Event heat: headlines in last 48h
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    heat = 0
    for it in news_items or []:
        ts = it.get('published')
        try:
            if ts and (now - ts) <= timedelta(hours=48):
                heat += 1
        except Exception:
            continue

    return {
        'opportunity_score': opportunity_score,
        'divergence_score': int(divergence_score),
        'risk_index': risk_index,
        'event_heat_48h': heat,
    }

def predict_trend(sentiment_score: float, price_history: pd.DataFrame | None = None) -> str:
    """Combine news sentiment with simple momentum for a short-term direction signal.

    Rules:
    - Momentum signal = sign(Close[-1] / SMA20[-1] - 1) if enough data, else 0
    - Sentiment signal = +1 if score>0.2, -1 if score<-0.2, else 0
    - Sum the two signals: >0 => Up, <0 => Down, else Stable
    """
    sentiment_signal = 1 if sentiment_score > 0.2 else (-1 if sentiment_score < -0.2 else 0)

    momentum_signal = 0
    if isinstance(price_history, pd.DataFrame) and not price_history.empty and 'Close' in price_history.columns:
        closes = price_history['Close'].dropna()
        if len(closes) >= 20:
            sma20 = closes.rolling(20).mean().iloc[-1]
            last = float(closes.iloc[-1])
            if np.isfinite(sma20) and sma20 > 0:
                rel = last / sma20 - 1.0
                momentum_signal = 1 if rel > 0 else (-1 if rel < 0 else 0)

    combined = sentiment_signal + momentum_signal
    if combined > 0:
        return "Up 📈"
    if combined < 0:
        return "Down 📉"
    return "Stable ↔️"


def _compute_indicators(price_history: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA20/50, RSI14, MACD(12,26,9), ATR14. Returns new DataFrame."""
    df = price_history.copy()
    if df.empty:
        return df
    close = df['Close']
    high = df['High']
    low = df['Low']

    # SMA (use strict min_periods to avoid early noise)
    df['SMA20'] = close.rolling(window=20, min_periods=20).mean()
    df['SMA50'] = close.rolling(window=50, min_periods=50).mean()
    df['SMA100'] = close.rolling(window=100, min_periods=100).mean()
    df['SMA200'] = close.rolling(window=200, min_periods=200).mean()

    # RSI14 (Wilder's)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd
    df['MACD_SIGNAL'] = signal

    # ATR14
    prev_close = close.shift(1)
    tr = np.maximum.reduce([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ])
    df['ATR14'] = pd.Series(tr, index=df.index).rolling(14).mean()
    return df


def advanced_predict_trend(sentiment_score: float, price_history: pd.DataFrame | None) -> dict:
    """Rule-based ensemble using sentiment + indicators. Returns direction, confidence, and explanations.

    Signals considered:
    - Sentiment: >0.4 bullish, <-0.4 bearish, else neutral
    - SMA20 vs SMA50: above bullish, below bearish
    - MACD above signal bullish; below bearish
    - RSI14: <30 oversold (bullish), >70 overbought (bearish)
    - Price vs SMA20 momentum
    Confidence scales with number and strength of aligned signals.
    """
    if price_history is None or not isinstance(price_history, pd.DataFrame) or price_history.empty:
        # fallback to simple predictor
        return {
            'direction': predict_trend(sentiment_score),
            'confidence': 50,
            'signals': ['Fallback: insufficient price history']
        }

    df = _compute_indicators(price_history)
    last = df.iloc[-1]
    signals: list[str] = []
    score = 0

    # Sentiment
    if sentiment_score > 0.8:
        score += 2; signals.append('Strong positive news sentiment')
    elif sentiment_score > 0.4:
        score += 1; signals.append('Positive news sentiment')
    elif sentiment_score < -0.8:
        score -= 2; signals.append('Strong negative news sentiment')
    elif sentiment_score < -0.4:
        score -= 1; signals.append('Negative news sentiment')
    else:
        signals.append('Neutral/mixed news sentiment')

    # SMA20 vs SMA50
    if pd.notna(last.get('SMA20')) and pd.notna(last.get('SMA50')) and pd.notna(last.get('Close')):
        if last['SMA20'] > last['SMA50'] and last['Close'] > last['SMA20']:
            score += 1; signals.append('Bullish: price above SMA20 and SMA20>SMA50')
        elif last['SMA20'] < last['SMA50'] and last['Close'] < last['SMA20']:
            score -= 1; signals.append('Bearish: price below SMA20 and SMA20<SMA50')

    # MACD
    if pd.notna(last.get('MACD')) and pd.notna(last.get('MACD_SIGNAL')):
        if last['MACD'] > last['MACD_SIGNAL']:
            score += 1; signals.append('MACD above signal (bullish)')
        elif last['MACD'] < last['MACD_SIGNAL']:
            score -= 1; signals.append('MACD below signal (bearish)')

    # RSI extremes
    if pd.notna(last.get('RSI14')):
        if last['RSI14'] < 30:
            score += 1; signals.append('RSI oversold (<30)')
        elif last['RSI14'] > 70:
            score -= 1; signals.append('RSI overbought (>70)')

    # Volatility awareness (ATR relative to price)
    if pd.notna(last.get('ATR14')) and pd.notna(last.get('Close')) and last['Close'] > 0:
        vol = float(last['ATR14'] / last['Close'])
        if vol > 0.04:
            signals.append('High volatility: reduce confidence')
            # dampen score magnitude
            score = np.sign(score) * max(0, abs(score) - 1)

    # Map score to direction
    if score > 0:
        direction = 'Up 📈'
    elif score < 0:
        direction = 'Down 📉'
    else:
        direction = 'Stable ↔️'

    # Confidence from |score|
    base_conf = {0: 40, 1: 60, 2: 75, 3: 85, 4: 92}.get(int(abs(score)), 95)
    confidence = int(base_conf)
    return {'direction': direction, 'confidence': confidence, 'signals': signals}


def get_trend_labels(price_history: pd.DataFrame) -> dict:
    """Return short-term and long-term trend labels from moving averages.

    Short-term: Close vs SMA20 and SMA20 vs SMA50
    Long-term: SMA50 vs SMA200
    """
    if not isinstance(price_history, pd.DataFrame) or price_history.empty:
        return {'short_term': 'Unknown', 'long_term': 'Unknown'}
    df = _compute_indicators(price_history)
    last = df.iloc[-1]
    short = 'Unknown'
    long = 'Unknown'
    try:
        # Short-term: prefer 20/50 structure; fallback to Close vs SMA20 when SMA50 missing
        has_close = pd.notna(last.get('Close'))
        has_sma20 = pd.notna(last.get('SMA20'))
        has_sma50 = pd.notna(last.get('SMA50'))
        if has_close and has_sma20 and has_sma50:
            if last['Close'] > last['SMA20'] and last['SMA20'] > last['SMA50']:
                short = 'Uptrend'
            elif last['Close'] < last['SMA20'] and last['SMA20'] < last['SMA50']:
                short = 'Downtrend'
            else:
                short = 'Sideways'
        elif has_close and has_sma20:
            # Fallback when SMA50 unavailable (short history)
            if last['Close'] > last['SMA20']:
                short = 'Uptrend'
            elif last['Close'] < last['SMA20']:
                short = 'Downtrend'
            else:
                short = 'Sideways'
        else:
            # Ultimate short-term fallback: recent 3-day change in Close
            closes = df['Close'].dropna()
            if len(closes) >= 3:
                delta3 = float(closes.iloc[-1] - closes.iloc[-3])
                if delta3 > 0:
                    short = 'Uptrend'
                elif delta3 < 0:
                    short = 'Downtrend'
                else:
                    short = 'Sideways'

        # Long-term preference: 50 vs 200, else 50 vs 100, else slope of 50
        sma50 = last.get('SMA50')
        sma100 = last.get('SMA100')
        sma200 = last.get('SMA200')
        if pd.notna(sma50) and pd.notna(sma200):
            if sma50 > sma200:
                long = 'Uptrend'
            elif sma50 < sma200:
                long = 'Downtrend'
            else:
                long = 'Sideways'
        elif pd.notna(sma50) and pd.notna(sma100):
            if sma50 > sma100:
                long = 'Uptrend'
            elif sma50 < sma100:
                long = 'Downtrend'
            else:
                long = 'Sideways'
        elif pd.notna(sma50):
            # Slope of SMA50 over last 5 days as fallback
            tail = df['SMA50'].dropna().tail(6)
            if len(tail) >= 2:
                slope = float(tail.iloc[-1] - tail.iloc[0])
                if slope > 0:
                    long = 'Uptrend'
                elif slope < 0:
                    long = 'Downtrend'
                else:
                    long = 'Sideways'
        else:
            # Ultimate long-term fallback: slope of Close over last up to 10 days
            closes = df['Close'].dropna().tail(10)
            if len(closes) >= 2:
                slope_close = float(closes.iloc[-1] - closes.iloc[0])
                if slope_close > 0:
                    long = 'Uptrend'
                elif slope_close < 0:
                    long = 'Downtrend'
                else:
                    long = 'Sideways'
    except Exception:
        pass
    # Ensure we never return 'Unknown'
    if short == 'Unknown':
        short = 'Sideways'
    if long == 'Unknown':
        long = 'Sideways'
    return {'short_term': short, 'long_term': long}


def _safe_div(numerator: float | int | None, denominator: float | int | None) -> float | None:
    try:
        if numerator is None or denominator in (None, 0):
            return None
        return float(numerator) / float(denominator)
    except Exception:
        return None


def get_fundamentals(ticker: str) -> dict:
    """Fetch fundamentals and compute basic valuation/quality ratios.

    Returns dict with keys: 'ratios', 'income_statement', 'balance_sheet', 'cash_flow', 'meta'.
    DataFrames are pandas objects suitable for display.
    """
    try:
        resolved = _resolve_ticker_symbol(ticker)
        t = yf.Ticker(resolved)

        # Core statements (annual)
        income = getattr(t, 'financials', pd.DataFrame())
        balance = getattr(t, 'balance_sheet', pd.DataFrame())
        cash = getattr(t, 'cashflow', pd.DataFrame())

        # Prefer most recent column
        def latest_value(df: pd.DataFrame, row_name: str) -> float | None:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None
            if row_name not in df.index:
                return None
            series = df.loc[row_name]
            if isinstance(series, pd.Series) and not series.empty:
                return float(series.iloc[0])
            return None

        # Prices / market data
        fast = getattr(t, 'fast_info', None)
        last_price = None
        market_cap = None
        if fast is not None:
            last_price = getattr(fast, 'last_price', None)
            market_cap = getattr(fast, 'market_cap', None)

        # Shares outstanding (approx)
        shares = None
        if market_cap and last_price and last_price != 0:
            shares = float(market_cap) / float(last_price)

        # Key line items
        revenue = latest_value(income, 'Total Revenue') or latest_value(income, 'Revenue')
        net_income = latest_value(income, 'Net Income') or latest_value(income, 'Net Income Common Stockholders')
        ebit = latest_value(income, 'Ebit') or latest_value(income, 'Operating Income')
        gross_profit = latest_value(income, 'Gross Profit')

        total_assets = latest_value(balance, 'Total Assets')
        total_liab = latest_value(balance, 'Total Liabilities Net Minority Interest') or latest_value(balance, 'Total Liab')
        total_equity = latest_value(balance, "Total Stockholder Equity") or (None if total_assets is None or total_liab is None else float(total_assets) - float(total_liab))
        current_assets = latest_value(balance, 'Total Current Assets')
        current_liab = latest_value(balance, 'Total Current Liabilities')
        total_debt = latest_value(balance, 'Total Debt') or latest_value(balance, 'Short Long Term Debt')

        cfo = latest_value(cash, 'Total Cash From Operating Activities') or latest_value(cash, 'Operating Cash Flow')
        capex = latest_value(cash, 'Capital Expenditures')
        fcf = None if cfo is None or capex is None else float(cfo) - float(capex)

        # Ratios
        pe = _safe_div(last_price, _safe_div(net_income, shares)) if last_price and shares and net_income else None
        ps = _safe_div(market_cap, revenue) if market_cap and revenue else None
        pb = _safe_div(market_cap, total_equity) if market_cap and total_equity else None
        roe = _safe_div(net_income, total_equity)
        roa = _safe_div(net_income, total_assets)
        debt_to_equity = _safe_div(total_debt, total_equity)
        current_ratio = _safe_div(current_assets, current_liab)
        ebit_margin = _safe_div(ebit, revenue)
        net_margin = _safe_div(net_income, revenue)
        fcf_margin = _safe_div(fcf, revenue) if fcf is not None else None

        ratios = {
            'Price': last_price,
            'Market Cap': market_cap,
            'P/E': pe,
            'P/S': ps,
            'P/B': pb,
            'ROE': roe,
            'ROA': roa,
            'Debt/Equity': debt_to_equity,
            'Current Ratio': current_ratio,
            'EBIT Margin': ebit_margin,
            'Net Margin': net_margin,
            'FCF Margin': fcf_margin,
        }

        # Tidy DataFrames for display: transpose so dates as rows
        def tidy(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return pd.DataFrame()
            tdf = df.copy()
            tdf = tdf.T
            tdf.index.name = 'Period'
            return tdf

        return {
            'ratios': ratios,
            'income_statement': tidy(income),
            'balance_sheet': tidy(balance),
            'cash_flow': tidy(cash),
            'meta': {'resolved_symbol': resolved}
        }
    except Exception as e:
        return {'error': f'Error fetching fundamentals: {e}'}


def get_current_price(ticker: str) -> float | None:
    """Return the latest trade price using fast_info; fallback to last close."""
    try:
        resolved = _resolve_ticker_symbol(ticker)
        t = yf.Ticker(resolved)
        last = None
        fast = getattr(t, 'fast_info', None)
        if fast is not None:
            last = getattr(fast, 'last_price', None)
        if last is None:
            hist = t.history(period="5d", interval="1d")
            if isinstance(hist, pd.DataFrame) and not hist.empty and 'Close' in hist.columns:
                last = float(hist['Close'].iloc[-1])
        return float(last) if last is not None else None
    except Exception:
        return None


def compute_uptrend_levels(price_history: pd.DataFrame) -> dict:
    """Compute heuristic uptrend trigger and near-term target.

    - Trigger: breakout above last 20-day high + 0.5% buffer
    - Near-term target: last close + 1 * ATR14 (momentum extension)
    """
    if not isinstance(price_history, pd.DataFrame) or price_history.empty:
        return {'error': 'No price history'}
    df = _compute_indicators(price_history)
    closes = df['Close'].dropna()
    if len(closes) < 21:
        return {'error': 'Not enough history for levels'}
    last_close = float(closes.iloc[-1])
    last_high20 = float(df['High'].rolling(20).max().iloc[-2]) if 'High' in df.columns else float(closes.rolling(20).max().iloc[-2])
    trigger = last_high20 * 1.005
    atr = None
    if 'ATR14' in df.columns and pd.notna(df['ATR14'].iloc[-1]):
        atr = float(df['ATR14'].iloc[-1])
    target = last_close + (atr if atr is not None else 0.03 * last_close)
    return {
        'last_close': last_close,
        'trigger_price': trigger,
        'near_term_target': target
    }


def backtest_price_strategy(price_history: pd.DataFrame) -> dict:
    """Backtest a simple, price-only strategy using our indicators.

    Rules (daily close-to-close):
    - Long when Close>SMA20 and SMA20>SMA50 and MACD>Signal
    - Short when Close<SMA20 and SMA20<SMA50 and MACD<Signal
    - Else neutral (flat)

    Metrics returned: hit_rate, total_return, cagr, max_drawdown, sharpe.
    Returns dict with 'metrics' and 'equity' (DataFrame with columns: Close, Position, StrategyRet, Equity).
    """
    if not isinstance(price_history, pd.DataFrame) or price_history.empty:
        return {'error': 'No price history'}

    df = _compute_indicators(price_history)
    df = df.dropna(subset=['Close']).copy()
    if df.empty:
        return {'error': 'No price history'}

    # Build position signal
    cond_long = (
        (df['Close'] > df['SMA20']) & (df['SMA20'] > df['SMA50']) & (df['MACD'] > df['MACD_SIGNAL'])
    )
    cond_short = (
        (df['Close'] < df['SMA20']) & (df['SMA20'] < df['SMA50']) & (df['MACD'] < df['MACD_SIGNAL'])
    )
    position = pd.Series(0, index=df.index, dtype=float)
    position[cond_long.fillna(False)] = 1.0
    position[cond_short.fillna(False)] = -1.0

    # Shift position by 1 to emulate taking trade on next bar
    df['Position'] = position.shift(1).fillna(0.0)

    # Returns
    df['CloseRet'] = df['Close'].pct_change().fillna(0.0)
    df['StrategyRet'] = df['Position'] * df['CloseRet']
    df['Equity'] = (1.0 + df['StrategyRet']).cumprod()

    # Metrics
    import math
    ret_series = df['StrategyRet']
    total_return = float(df['Equity'].iloc[-1] - 1.0)
    # Approximate trading days per year
    n = len(df)
    years = max(1e-9, n / 252.0)
    cagr = float(df['Equity'].iloc[-1] ** (1.0 / years) - 1.0) if n > 1 else 0.0
    # Max drawdown
    roll_max = df['Equity'].cummax()
    drawdown = df['Equity'] / roll_max - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    # Sharpe (daily to annual, risk-free ~0)
    mean_daily = float(ret_series.mean())
    std_daily = float(ret_series.std(ddof=0))
    sharpe = float((mean_daily / (std_daily if std_daily != 0 else math.inf)) * math.sqrt(252)) if std_daily > 0 else 0.0
    # Hit rate: direction vs next day return
    hits = (df['Position'] * df['CloseRet'] > 0).sum()
    total_trades = (df['Position'].diff().abs() > 0).sum()
    hit_rate = float(hits) / float(max(1, total_trades))

    metrics = {
        'hit_rate': round(hit_rate * 100, 1),
        'total_return_pct': round(total_return * 100, 1),
        'cagr_pct': round(cagr * 100, 1),
        'max_drawdown_pct': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 2),
        'bars': n,
        'trades': int(total_trades),
    }

    return {'metrics': metrics, 'equity': df[['Close', 'Position', 'StrategyRet', 'Equity']]}


def monte_carlo_forecast(price_history: pd.DataFrame, days_ahead: int = 252, sims: int = 500, seed: int | None = None, override_mu_ann: float | None = None, override_sigma_ann: float | None = None) -> dict:
    """Monte Carlo forecast using geometric Brownian motion estimated from historical log returns.

    Returns dict with 'paths' (DataFrame of simulated prices), 'percentiles' (DataFrame with p10/p50/p90), and 'params'.
    """
    if not isinstance(price_history, pd.DataFrame) or price_history.empty:
        return {'error': 'No price history'}
    closes = price_history['Close'].dropna().astype(float)
    if len(closes) < 30:
        return {'error': 'Not enough history for forecast'}

    import numpy as np
    rng = np.random.default_rng(seed)
    log_rets = np.log(closes / closes.shift(1)).dropna()
    mu = float(log_rets.mean()) * 252.0  # annualized drift
    sigma = float(log_rets.std(ddof=0)) * np.sqrt(252.0)  # annualized vol
    if override_mu_ann is not None:
        mu = float(override_mu_ann)
    if override_sigma_ann is not None:
        sigma = float(override_sigma_ann)
    dt = 1.0 / 252.0
    last_price = float(closes.iloc[-1])

    # Simulate GBM paths
    steps = days_ahead
    shocks = rng.normal((mu - 0.5 * sigma * sigma) * dt, sigma * np.sqrt(dt), size=(steps, sims))
    # cumulative log returns per path
    cum = shocks.cumsum(axis=0)
    paths = last_price * np.exp(cum)
    # prepend the current price
    paths = np.vstack([np.full((1, sims), last_price), paths])

    # Build indices (business days approx)
    import pandas as pd
    idx = pd.RangeIndex(start=0, stop=steps + 1, step=1)
    paths_df = pd.DataFrame(paths, index=idx)

    # Percentile bands per future day
    p10 = np.percentile(paths, 10, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p90 = np.percentile(paths, 90, axis=1)
    percentiles = pd.DataFrame({'p10': p10, 'p50': p50, 'p90': p90}, index=idx)

    return {
        'paths': paths_df,
        'percentiles': percentiles,
        'params': {'mu_ann': mu, 'sigma_ann': sigma, 'last_price': last_price}
    }