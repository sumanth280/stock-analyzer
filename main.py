import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from transformers import pipeline

# Load a pre-trained sentiment analysis model
# Note: This is a large model, so the first run will take time to download.
# A GPU is highly recommended for faster inference.
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

def get_stock_data(ticker):
    """
    Fetches historical stock data for a given ticker.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        return hist
    except Exception as e:
        return f"Error fetching stock data: {e}"

def get_financial_news(ticker):
    """
    Scrapes recent financial news headlines for a given ticker.
    This is a generic example and may require adjustments for specific websites.
    """
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        headlines = []
        # Find news headlines based on the website's HTML structure
        news_elements = soup.find_all('h3', class_='Mb(5px)')
        for element in news_elements:
            headlines.append(element.text.strip())
        
        return headlines
    except Exception as e:
        return f"Error scraping news: {e}"

def analyze_sentiment(text_list):
    """
    Analyzes the sentiment of a list of texts using the pre-trained model.
    """
    if not text_list:
        return {'sentiment_score': 0, 'label': 'Neutral'}
    
    results = sentiment_analyzer(text_list)
    
    # Calculate a combined sentiment score
    pos_score = sum(r['score'] for r in results if r['label'] == 'POSITIVE')
    neg_score = sum(r['score'] for r in results if r['label'] == 'NEGATIVE')
    
    sentiment_score = pos_score - neg_score
    
    if sentiment_score > 0.5:
        final_label = 'Strongly Positive'
    elif sentiment_score > 0:
        final_label = 'Positive'
    elif sentiment_score < -0.5:
        final_label = 'Strongly Negative'
    elif sentiment_score < 0:
        final_label = 'Negative'
    else:
        final_label = 'Neutral'
        
    return {'sentiment_score': sentiment_score, 'label': final_label}

def predict_trend(sentiment_score):
    """
    Predicts a short-term trend based on the sentiment score.
    Note: This is a simplified, rule-based prediction. Real-world stock prediction
    is highly complex and involves advanced machine learning models.
    """
    if sentiment_score > 0.5:
        return "Up 📈"
    elif sentiment_score < -0.5:
        return "Down 📉"
    else:
        return "Stable ↔️"