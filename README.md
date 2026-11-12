# Local Volatility Model for Equity Option Pricing

Dupire local volatility model built from an implied volatility smile, with European option pricing by finite-difference PDE, then extended to real S&P 500 option quotes.

## Contents

| File | Description |
|---|---|
| `local_vol_model.ipynb` | Main notebook: synthetic SVI smile, Black-Scholes price surface, Dupire local volatility by finite differences, implicit PDE pricer, round-trip test, and a summary of the tests on market data. |
| `local_vol_market.py` | Calibration to market quotes (`^SPX` via yfinance, or a CSV): forward and discount factor from put-call parity, SVI fit per maturity with linear wings, analytic local volatility (Gatheral), log-price Crank-Nicolson PDE, round-trip test against bid-ask, and a comparison with the finite-difference approach. |

## Key results

- **Synthetic smile**: the local vol model reprices the Black-Scholes smile prices with a maximum absolute error of about 0.011 (at the money), against up to 1.17 for the smile effect itself.
- **SPX-like synthetic chain**: median round-trip error of about 2 vol bp (11 bp max), 96% of prices inside the bid-ask spread.
- **Finite differences on prices** are unstable at short maturities (80 to 88% of the grid unusable below 0.03 years), which motivates the analytic, implied-volatility form of Dupire's formula.

## Usage

```bash
pip install -r requirements.txt
jupyter notebook local_vol_model.ipynb
python local_vol_market.py          # set USE_MARKET = False to run on a synthetic chain
```

Market data: run during US market hours (15:30–22:00 Paris time); outside them, yfinance quotes are stale or empty. Use European options only (e.g. `^SPX`).

## Methods

- Raw SVI smile, within Lee's moment bound
- Dupire's formula with a continuous dividend yield; Gatheral's formula in total implied variance
- Implicit and Crank-Nicolson (Rannacher smoothing) finite-difference schemes
- Implied forward and discount curve from put-call parity, with robust regression

## Author

Chris Teku
