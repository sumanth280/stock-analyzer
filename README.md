# 📊 AI News Sentiment & Stock Analyzer (US + India)

Analyze financial news with **FinBERT**, blend it with technicals, fundamentals, and unique alpha signals, and visualize cleanly in **Streamlit**.  
Supports **U.S. and Indian stocks** (auto‑resolves NSE `.NS` / BSE `.BO`).  

🌐 **[Live Demo](https://stock-analyzer99.streamlit.app/)**  

---

## 👨‍💻 Developed by: *Sumanth Nerella*  

---

## 🚀 What’s Inside  

### 📰 Sentiment (FinBERT)  
- Overall label and score; per‑headline labels/scores  
- Yahoo Finance headlines with Google News RSS fallback  

### 📈 Prediction  
- Rule‑based short‑term signal combining sentiment + SMA20 momentum  
- Advanced predictor adds RSI14, MACD, ATR and outputs direction + confidence + rationale  

### 📊 Charting  
- Interactive candlestick with selectable history: 1mo, 3mo, 6mo, 1y, 3y, 5y  
- Current Price, Uptrend Trigger (20‑day breakout), Near‑term Target (ATR‑based)  

### 💰 Fundamentals  
- Key ratios: Price, Market Cap, P/E, P/S, P/B, ROE, ROA, Debt/Equity, Current Ratio, EBIT/Net/FCF Margins  
- Financial statements (annual): Income Statement, Balance Sheet, Cash Flow  
- Plain‑language glossary for metrics  

### 🔮 Alpha Signals (unique)  
- **Opportunity Score (0–100):** blends sentiment strength, RSI extreme proximity, MACD cross proximity  
- **Divergence:** sentiment vs last 5‑day return (captures price–news disconnects)  
- **Risk Index:** ATR as % of price (volatility awareness)  
- **News Heat (48h):** recent headline count  

### 🎨 UX  
- Sidebar controls, wide layout, neat styling  
- Data persists across tabs (no re‑clicking Analyze)  
- Explanatory tooltips and glossaries for non‑experts  

---

## ⚡ Quickstart  

```bash
# 1) Python 3.10+
# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

👉 In the app:  
- Enter a stock (e.g., RELIANCE, TCS, AAPL, TSLA)  
- Choose a period (1mo–5y)  
- Click **Analyze** once; switch tabs freely  

---

## 🧠 How Predictions Work  

- **Sentiment score** = sum(POSITIVE) − sum(NEGATIVE) across headlines using FinBERT tone  
- **Simple signal:** sentiment threshold + price above/below SMA20  
- **Advanced signal:** adds MACD vs signal, SMA20/50 slope, RSI extremes, ATR volatility  
- **Outputs:** Up/Down/Stable + confidence + human‑readable reasons  

---

## 📌 Uptrend Trigger & Target  

- **Trigger:** breakout above previous 20‑day high with a small buffer (~0.5%)  
- **Target:** last close + 1×ATR14 (momentum extension heuristic)  

---

## 🏦 Fundamentals Data  

- Pulled from yfinance’s financials/balance_sheet/cashflow/fast_info  
- Ratios computed when possible; missing fields shown as —  

---

## 🇮🇳 Indian Stock Support  

- Enter raw symbol (e.g., RELIANCE, TCS).  
- App tries variants: raw, `.NS`, `.BO` and uses the first with data.  

---

## ⚙️ Caching & Performance  

- News, prices, fundamentals cached for ~10 minutes  
- First FinBERT run downloads the model (~hundreds of MB)  

---

## 🚀 Deploying  

- **Streamlit Community Cloud:** point to this repo, main file `app.py`  
- **Hugging Face Spaces (Streamlit):** include `requirements.txt`  
- **Render/Railway:**  
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 🛠 Troubleshooting  

- **Missing package:** `pip install -r requirements.txt` (or install the named package)  
- **Model download slow:** first run only; subsequent runs are faster  
- **No Indian data:** try `RELIANCE.NS` / `TCS.BO` or another stock  

---

## 🏗 Tech Stack  

- Streamlit, Plotly, Pandas, NumPy, yfinance  
- Transformers (FinBERT tone model), feedparser  

---

## ⚠️ Disclaimer  

This is for **educational purposes only** and not financial advice.  

---

## ✨ What’s Unique (Interview Highlights)  

- Practical blend of **news sentiment (FinBERT) with technical indicators** to create explainable signals, not a black box.  
- Original **“Alpha Signals” dashboard**: Opportunity Score, News–Price Divergence, Risk Index (ATR%), and News Heat.  
- **Auto-resolution for Indian tickers (NSE/BSE)** plus robust news fallback via Google RSS.  
- Clear **glossaries/tooltips for non‑experts**, improving usability beyond typical stock tools.  
