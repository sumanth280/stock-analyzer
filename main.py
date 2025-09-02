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