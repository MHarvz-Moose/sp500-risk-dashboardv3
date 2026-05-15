# Multi-Asset 3–6 Month Decision Support Dashboard — Optimized Build

Assesses 3–6 month holding attractiveness, drawdown risk, drawdown severity, latent fragility, and primary drivers across S&P 500, Gold, Oil, Bitcoin, FTSE All-World, and FTSE Emerging Markets.

Uses yfinance, FRED, and structured manual inputs.

## Deployment

Upload the contents of this folder to GitHub. In Streamlit Cloud set:

```text
Main file path: app.py
```

Add your FRED key under Settings > Secrets:

```toml
FRED_API_KEY = "your_actual_fred_key_here"
```

Do not upload your real FRED key to GitHub.
