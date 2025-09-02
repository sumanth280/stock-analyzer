AI News Sentiment & Stock Predictor

Analyze financial news with FinBERT and combine with simple price momentum to estimate short‑term trend. Supports U.S. and Indian stocks (auto-resolves NSE/BSE suffixes).

Features
- FinBERT sentiment on latest headlines
- Momentum + sentiment short‑term direction (Up/Down/Stable)
- Candlestick chart (last 3 months)
- Works for tickers like `AAPL`, `TSLA`, `RELIANCE` (NSE), `TCS` (NSE)

Setup
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run app.py
```

Notes
- First run downloads the FinBERT model (~400MB); allow time.
- For Indian equities, raw symbols will be tried with `.NS` and `.BO` automatically.

Disclaimer
This is for educational purposes only and not financial advice.

