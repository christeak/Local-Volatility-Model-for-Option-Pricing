"""
Dupire local volatility calibrated to market option prices.

Pipeline:
  1. Load an option chain (yfinance or CSV)
  2. Clean it (bid/ask, spreads)
  3. Implied forward F(T) and discount factor D(T) from put-call parity
     -> no need to know r or q
  4. Implied vols of out-of-the-money options (puts for K < F, calls for K >= F)
  5. One raw SVI slice per maturity. Beyond the quoted strikes, total variance is
     extended linearly within Lee's bound, so the wings cannot explode.
     Arbitrage is checked, not imposed.
  6. Analytic local volatility (Gatheral's formula), no finite differences in K
  7. PDE pricer in log-price, grid sized to each maturity, Crank-Nicolson
     with Rannacher smoothing, time-dependent rates and dividends
  8. Round-trip test: reprice the market quotes with the local vol model

Use EUROPEAN options (e.g. ^SPX, ^STOXX50E). Options on single stocks and ETFs
(SPY, AAPL...) are American: put-call parity and Black's formula do not hold exactly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq, least_squares
from scipy.linalg import solve_banded
from scipy.interpolate import RegularGridInterpolator


# =====================================================================
# 1. Data loading
# =====================================================================

def load_chain_yfinance(ticker="^SPX", n_expiries=8, min_days=14, max_days=730):
    """
    Option chain from yfinance. Returns (spot, DataFrame) with columns
    expiry, T, strike, type ('call'/'put'), bid, ask, volume, openInterest, root.
    Run it during US market hours (15:30-22:00 Paris time): outside them, quotes are
    stale or bid = ask = 0, and the spot may not match the option quotes.
    """
    import yfinance as yf
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    now = pd.Timestamp.now(tz="America/New_York")

    def year_fraction(expiry):
        # PM-settled expiry at 16:00 New York time; precise T matters a lot for short maturities
        close = pd.Timestamp(expiry).tz_localize("America/New_York") + pd.Timedelta(hours=16)
        return (close - now).total_seconds() / (365.0 * 24 * 3600)

    expiries = [(e, year_fraction(e)) for e in tk.options]
    expiries = [(e, T) for e, T in expiries if min_days / 365 <= T <= max_days / 365]
    if not expiries:
        raise ValueError("No expiry in the requested range.")
    # keep n_expiries expiries spread over the range
    idx = np.unique(np.linspace(0, len(expiries) - 1, n_expiries).round().astype(int))
    expiries = [expiries[i] for i in idx]

    frames = []
    for e, T in expiries:
        chain = tk.option_chain(e)
        for df, opt_type in ((chain.calls, "call"), (chain.puts, "put")):
            d = df[["contractSymbol", "strike", "bid", "ask", "volume", "openInterest"]].copy()
            # root = letters before the date, e.g. SPX (AM-settled monthly) or SPXW (PM-settled)
            d["root"] = d["contractSymbol"].str.extract(r"^([A-Z]+)")[0]
            d["type"], d["T"], d["expiry"] = opt_type, T, e
            frames.append(d.drop(columns="contractSymbol"))
    return spot, pd.concat(frames, ignore_index=True)


def load_chain_csv(path, spot):
    """CSV with at least: T (in years) or expiry (date), strike, type, bid, ask."""
    df = pd.read_csv(path)
    if "T" not in df.columns:
        today = pd.Timestamp.today().normalize()
        df["T"] = (pd.to_datetime(df["expiry"]) - today).dt.days / 365.0
    df["type"] = df["type"].str.lower().str[0].map({"c": "call", "p": "put"})
    return spot, df


# =====================================================================
# 2. Cleaning
# =====================================================================

def clean_chain(df, max_rel_spread=0.5):
    df = df.copy()
    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"])]
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"]
    df = df[df["rel_spread"] < max_rel_spread]
    # the same expiry can be listed under two roots (e.g. SPX and SPXW): keep one quote
    # per (T, strike, type), the one with the tightest spread
    df = df.sort_values("rel_spread").drop_duplicates(["T", "strike", "type"], keep="first")
    return df.sort_values(["T", "strike"]).reset_index(drop=True)


# =====================================================================
# 3. Forward and discount factor from put-call parity
# =====================================================================

def forward_and_discount(df, spot, band=0.15, r_bounds=(-0.02, 0.15)):
    """
    For each maturity, C - P = D (F - K).
      1. Robust regression of C - P on K near the money: slope -D, intercept D·F.
      2. The discount curve is smooth, so ln D(T) = -(r0 + r1 T) T is fitted robustly
         across maturities; slices with an implausible rate get the fitted D instead.
      3. Given D, each forward is the median of K + (C - P) / D over near-the-money strikes.
    """
    raw = []
    for T, g in df.groupby("T"):
        calls = g[g["type"] == "call"].groupby("strike")["mid"].mean()
        puts = g[g["type"] == "put"].groupby("strike")["mid"].mean()
        K = calls.index.intersection(puts.index)
        K = K[np.abs(np.log(K / spot)) < band]
        if len(K) < 5:
            continue
        Kv, y = K.values.astype(float), (calls[K] - puts[K]).values
        scale = max(np.median(np.abs(y)), 1.0)
        fit = least_squares(lambda p: (p[0] + p[1] * Kv - y) / scale,
                            np.polyfit(Kv, y, 1)[::-1], loss="soft_l1")
        D = -fit.x[1]
        raw.append({"T": T, "D_raw": D, "pairs": (Kv, y)})
    if len(raw) < 2:
        raise ValueError("Cannot estimate forwards (not enough call/put pairs).")

    Ts = np.array([x["T"] for x in raw])
    D_raw = np.array([x["D_raw"] for x in raw])
    r_raw = -np.log(np.clip(D_raw, 1e-6, None)) / Ts
    ok = (r_raw > r_bounds[0]) & (r_raw < r_bounds[1])
    if ok.sum() >= 2:
        w = np.sqrt(Ts[ok])     # long maturities pin the rate down much better
        fit = least_squares(lambda p: w * (p[0] + p[1] * Ts[ok] - r_raw[ok]),
                            [np.median(r_raw[ok]), 0.0], loss="soft_l1", f_scale=0.005)
        r0, r1 = fit.x
    else:
        r0, r1 = np.median(np.clip(r_raw, *r_bounds)), 0.0
    r_fit = r0 + r1 * Ts

    out = []
    for x, T, rr, rf, good in zip(raw, Ts, r_raw, r_fit, ok):
        # keep the slice's own rate only if it is plausible and close to the smooth curve
        r_used = rr if (good and abs(rr - rf) < 0.01) else rf
        D = np.exp(-r_used * T)
        Kv, y = x["pairs"]
        F = np.median(Kv + y / D)
        out.append({"T": T, "D": D, "F": F, "r": r_used,
                    "q": r_used - np.log(F / spot) / T, "r_raw": rr, "replaced": r_used != rr})
    return pd.DataFrame(out)


def make_curves(curve, spot):
    """D(t) and F(t): linear interpolation of ln D and ln F, with D(0) = 1, F(0) = spot."""
    Ts = np.r_[0.0, curve["T"].values]
    lnD = np.r_[0.0, np.log(curve["D"].values)]
    lnF = np.r_[np.log(spot), np.log(curve["F"].values)]

    def _interp(t, y):
        t = np.asarray(t, float)
        # linear extrapolation beyond the last maturity (constant rates)
        slope = (y[-1] - y[-2]) / (Ts[-1] - Ts[-2])
        return np.where(t <= Ts[-1], np.interp(t, Ts, y), y[-1] + slope * (t - Ts[-1]))

    D = lambda t: np.exp(_interp(t, lnD))
    F = lambda t: np.exp(_interp(t, lnF))
    return D, F


# =====================================================================
# 4. Implied volatility (Black's formula on the forward)
# =====================================================================

def black_price(F, K, T, D, sigma, option_type):
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s**2) / s
    d2 = d1 - s
    if option_type == "call":
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, D, option_type):
    intrinsic = D * max(F - K, 0) if option_type == "call" else D * max(K - F, 0)
    if not np.isfinite(price) or price <= intrinsic + 1e-10:
        return np.nan
    f = lambda s: black_price(F, K, T, D, s, option_type) - price
    try:
        return brentq(f, 1e-4, 5.0)
    except ValueError:
        return np.nan


def otm_quotes(df, curve, max_abs_k=0.6):
    """Keep out-of-the-money options and compute their implied vols (mid, bid, ask)."""
    rows = []
    for _, c in curve.iterrows():
        g = df[np.isclose(df["T"], c["T"])]
        for _, o in g.iterrows():
            K, opt_type = o["strike"], o["type"]
            k = np.log(K / c["F"])
            if abs(k) > max_abs_k:
                continue
            if (opt_type == "call" and K < c["F"]) or (opt_type == "put" and K >= c["F"]):
                continue
            iv = implied_vol(o["mid"], c["F"], K, c["T"], c["D"], opt_type)
            if np.isnan(iv):
                continue
            rows.append({
                "T": c["T"], "strike": K, "type": opt_type, "k": k, "mid": o["mid"], "iv": iv,
                "iv_bid": implied_vol(o["bid"], c["F"], K, c["T"], c["D"], opt_type),
                "iv_ask": implied_vol(o["ask"], c["F"], K, c["T"], c["D"], opt_type),
            })
    return pd.DataFrame(rows)


# =====================================================================
# 5. SVI surface: one raw SVI slice per maturity, linear wings
# =====================================================================
#
#   w_i(k) = a + b [ρ (k - m) + sqrt((k - m)^2 + s^2)],   k = ln(K / F(T_i))
#
#   Each slice is fitted independently on its quotes. Beyond the last quoted strike
#   on each side, the smile is NOT the SVI extrapolation: total variance is extended
#   linearly from the edge of the quoted range, with a slope kept within Lee's bound
#   (|∂w/∂k| <= 2, and the right/left wing cannot go down/up). Arbitrage is not imposed
#   but checked afterwards (check_arbitrage).

def svi_raw(k, p):
    a, b, rho, m, s = p
    x = k - m
    R = np.sqrt(x**2 + s**2)
    return a + b * (rho * x + R), b * (rho + x / R), b * s**2 / R**3   # w, w', w''


def gatheral_g(y, w, wy, wyy):
    return (1 - y * wy / (2 * w)) ** 2 - 0.25 * wy**2 * (1 / w + 0.25) + 0.5 * wyy


def _quasi_explicit_guess(k, w_mkt, wts):
    """
    Zeliade's quasi-explicit idea: for fixed (m, s), w = a + c (k - m) + d sqrt((k - m)^2 + s^2)
    is linear in (a, c, d), with c = bρ and d = b. A weighted linear least squares on a grid
    of (m, s) gives a near-global starting point for the non-linear fit.
    """
    best, best_err = None, np.inf
    for m in np.linspace(-0.3, 0.3, 13):
        for s in np.geomspace(1e-3, 1.0, 16):
            x = k - m
            R = np.sqrt(x**2 + s**2)
            A = np.column_stack([np.ones_like(k), x, R]) * wts[:, None]
            coef, *_ = np.linalg.lstsq(A, w_mkt * wts, rcond=None)
            a, c, d = coef
            if d <= 1e-6 or abs(c) >= d:          # need b > 0 and |ρ| < 1
                continue
            err = np.sum((A @ coef - w_mkt * wts) ** 2)
            if err < best_err:
                best, best_err = np.array([a, d, c / d, m, s]), err
    return best


def fit_svi_slice(k, iv, T, weights):
    """Raw SVI fit of one slice; residuals in implied vol, positive minimum variance."""
    w_mkt = iv**2 * T

    def resid(p):
        a, b, rho, m, s = p
        w = np.maximum(svi_raw(k, p)[0], 1e-12)
        min_var = a + b * s * np.sqrt(1 - rho**2)
        return np.r_[weights * (np.sqrt(w / T) - iv), 10 * max(0.0, -min_var) / w_mkt.mean()]

    lb = np.array([-w_mkt.max(), 1e-5, -0.999, -1.0, 1e-3])
    ub = np.array([w_mkt.max(), 4.0, 0.999, 1.0, 2.0])
    starts = []
    guess = _quasi_explicit_guess(k, w_mkt, weights / (2 * iv * T))   # vol errors -> w errors
    if guess is not None:
        starts.append(guess)
    atm_w = max(np.interp(0.0, np.sort(k), w_mkt[np.argsort(k)]), 1e-6)
    for rho0, s0 in [(-0.7, 0.1), (-0.3, 0.3)]:  # generic fallbacks
        b0 = 0.5 * atm_w / s0
        starts.append(np.array([atm_w - b0 * s0, b0, rho0, 0.0, s0]))

    best = None
    for x0 in starts:
        res = least_squares(resid, np.clip(x0, lb + 1e-9, ub - 1e-9), bounds=(lb, ub))
        if best is None or res.cost < best.cost:
            best = res
    return best.x


class SVISlice:
    """SVI inside the quoted range [k_lo, k_hi], linear total variance outside."""

    def __init__(self, T, params, k_lo, k_hi):
        self.T, self.params, self.k_lo, self.k_hi = T, np.asarray(params, float), k_lo, k_hi
        w_lo, s_lo, _ = svi_raw(np.array([k_lo]), self.params)
        w_hi, s_hi, _ = svi_raw(np.array([k_hi]), self.params)
        self.w_lo, self.w_hi = float(w_lo[0]), float(w_hi[0])
        self.s_lo = float(np.clip(s_lo[0], -2.0, 0.0))   # left wing: non-increasing in k
        self.s_hi = float(np.clip(s_hi[0], 0.0, 2.0))    # right wing: non-decreasing in k

    def derivatives(self, k):
        """w, ∂w/∂k, ∂²w/∂k²."""
        k = np.asarray(k, float)
        w, wk, wkk = svi_raw(k, self.params)
        left, right = k < self.k_lo, k > self.k_hi
        w = np.where(left, self.w_lo + self.s_lo * (k - self.k_lo), w)
        w = np.where(right, self.w_hi + self.s_hi * (k - self.k_hi), w)
        wk = np.where(left, self.s_lo, np.where(right, self.s_hi, wk))
        wkk = np.where(left | right, 0.0, wkk)
        return np.maximum(w, 1e-12), wk, wkk


class SVISurface:
    """
    Total variance w(y, t), y = ln(K / F(t)). Between two fitted maturities, w is
    interpolated linearly in t at fixed y; before the first one, w = w_1 · t / T_1;
    beyond the last one, it is extrapolated with the last slope.
    """

    def __init__(self, slices):
        self.slices = slices
        self.T_nodes = np.array([sl.T for sl in slices])

    def derivatives(self, y, t):
        """w, ∂w/∂y, ∂²w/∂y², ∂w/∂t at fixed y."""
        Ts = self.T_nodes
        if t <= Ts[0]:
            w1, w1y, w1yy = self.slices[0].derivatives(y)
            lam = t / Ts[0]
            return lam * w1, lam * w1y, lam * w1yy, w1 / Ts[0]
        j = int(min(np.searchsorted(Ts, t) - 1, len(Ts) - 2))
        lam = (t - Ts[j]) / (Ts[j + 1] - Ts[j])
        w0, w0y, w0yy = self.slices[j].derivatives(y)
        w1, w1y, w1yy = self.slices[j + 1].derivatives(y)
        return ((1 - lam) * w0 + lam * w1, (1 - lam) * w0y + lam * w1y,
                (1 - lam) * w0yy + lam * w1yy, (w1 - w0) / (Ts[j + 1] - Ts[j]))

    def w(self, y, t):
        return np.maximum(self.derivatives(y, t)[0], 1e-12)

    def implied_vol(self, k, T):
        k, T = np.broadcast_arrays(np.asarray(k, float), np.asarray(T, float))
        out = np.empty(k.shape)
        for t in np.unique(T):
            sel = T == t
            out[sel] = np.sqrt(self.w(k[sel], t) / t)
        return out


def fit_surface(quotes, min_points=6):
    q = quotes.groupby("T").filter(lambda g: len(g) >= min_points)
    if q["T"].nunique() < 2:
        raise ValueError("At least two usable maturities are required.")
    slices = []
    for T, g in q.groupby("T"):
        spread = (g["iv_ask"] - g["iv_bid"]).values
        fill = np.nanmedian(spread) if np.isfinite(spread).any() else 0.01
        spread = np.where(np.isfinite(spread), spread, fill)
        weights = 1.0 / np.clip(spread, 0.002, 0.05)   # tighter spread -> more weight
        weights /= weights.mean()
        k = g["k"].values
        p = fit_svi_slice(k, g["iv"].values, T, weights)
        slices.append(SVISlice(T, p, k.min(), k.max()))
    return SVISurface(slices), q


def check_arbitrage(surface, k=np.linspace(-1.0, 1.0, 201)):
    """Butterfly g(k) >= 0 on each slice, calendar w increasing in T; warns if violated."""
    w_prev = None
    for sl in surface.slices:
        w, wk, wkk = sl.derivatives(k)
        g = gatheral_g(k, w, wk, wkk)
        if (g < 0).any():
            bad = k[g < 0]
            print(f"  [!] butterfly arbitrage at T = {sl.T:.3f}, k in [{bad.min():.2f}, {bad.max():.2f}]")
        if w_prev is not None and (w < w_prev).any():
            bad = k[w < w_prev]
            print(f"  [!] calendar arbitrage before T = {sl.T:.3f}, k in [{bad.min():.2f}, {bad.max():.2f}]")
        w_prev = w


# =====================================================================
# 6. Local volatility (Gatheral's formula)
# =====================================================================

def make_local_vol(surface, F, vmin=0.01, vmax=3.0):
    """sigma_loc(S, t)^2 = ∂w/∂t / g(y), with y = ln(S / F(t)) held fixed."""
    def sigma(S, t):
        t = max(t, 1e-4)
        y = np.log(S / F(t))
        w, wy, wyy, wt = surface.derivatives(y, t)
        w = np.maximum(w, 1e-12)
        g = gatheral_g(y, w, wy, wyy)
        # where the surface is arbitrageable, fall back on the implied variance
        var = np.where((g > 1e-10) & (wt > 0), wt / np.maximum(g, 1e-10), w / t)
        return np.clip(np.sqrt(var), vmin, vmax)

    return sigma


# =====================================================================
# 7. PDE pricer in x = ln S (Crank-Nicolson + Rannacher smoothing)
# =====================================================================
#
#   ∂V/∂t + (μ(t) - σ²/2) ∂V/∂x + σ²/2 ∂²V/∂x² - r(t) V = 0,
#   with r(t) = -d ln D/dt and μ(t) = r - q = d ln F/dt.

def _half_width(surface, T, side, n_std=6.0):
    """Half-width L of the log grid on one side: L = n_std · sqrt(w(±L, T)), solved by iteration."""
    L = n_std * np.sqrt(max(surface.w(0.0, T), 1e-8))
    for _ in range(30):
        L = n_std * np.sqrt(max(surface.w(side * L, T), 1e-8))
    return float(np.clip(L, 0.1, 4.0))


def price_local_vol(S0, K, T, option_type, sigma, D, F, surface,
                    M=500, N=None, n_std=6.0, rannacher_steps=2):
    if N is None:
        N = int(np.clip(400 * np.sqrt(T), 80, 600))

    # --- grid in x = ln S, sized to the distribution of S_T ---
    lnF_T, lnS0, lnK = np.log(F(T)), np.log(S0), np.log(K)
    x_lo = min(lnF_T, lnS0) - _half_width(surface, T, -1, n_std)
    x_hi = max(lnF_T, lnS0) + _half_width(surface, T, +1, n_std)
    x_lo, x_hi = min(x_lo, lnK - 0.05), max(x_hi, lnK + 0.05)
    dx = (x_hi - x_lo) / M
    x_lo = lnK - np.ceil((lnK - x_lo) / dx) * dx    # put the strike exactly on a node
    x = x_lo + dx * np.arange(M + 1)
    S = np.exp(x)
    Si = S[1:-1]

    DT, FT = D(T), F(T)

    def boundaries(t):
        disc, fwd = DT / D(t), FT / F(t)
        if option_type == "call":
            return 0.0, disc * (S[-1] * fwd - K)
        return disc * (K - S[0] * fwd), 0.0

    V = np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)

    # time steps from T back to 0; the first ones are split into implicit half-steps
    dt = T / N
    steps = [(dt / 2, 1.0)] * (2 * rannacher_steps) + [(dt, 0.5)] * (N - rannacher_steps)

    t_old = T
    for h, th in steps:
        t_new = max(t_old - h, 0.0)
        h = t_old - t_new
        if h <= 0:
            break
        r_t = -(np.log(D(t_old)) - np.log(D(t_new))) / h
        mu_t = (np.log(F(t_old)) - np.log(F(t_new))) / h
        sig2 = sigma(Si, 0.5 * (t_new + t_old)) ** 2
        alpha = 0.5 * sig2 / dx**2
        beta = (mu_t - 0.5 * sig2) / (2 * dx)
        a, b, c = alpha - beta, -2 * alpha - r_t, alpha + beta

        lo_new, up_new = boundaries(t_new)
        rhs = V[1:-1] + (1 - th) * h * (a * V[:-2] + b * V[1:-1] + c * V[2:])
        rhs[0] += th * h * a[0] * lo_new
        rhs[-1] += th * h * c[-1] * up_new

        ab = np.zeros((3, M - 1))
        ab[0, 1:] = -th * h * c[:-1]
        ab[1, :] = 1 - th * h * b
        ab[2, :-1] = -th * h * a[1:]
        V[1:-1] = solve_banded((1, 1), ab, rhs)
        V[0], V[-1] = lo_new, up_new
        t_old = t_new

    return float(np.interp(lnS0, x, V))


# =====================================================================
# 8. Round-trip test
# =====================================================================

def round_trip(quotes, spot, sigma, D, F, surface, per_slice=9):
    rows = []
    for T, g in quotes.groupby("T"):
        g = g.sort_values("k")
        sub = g.iloc[np.unique(np.linspace(0, len(g) - 1, per_slice).round().astype(int))]
        for _, o in sub.iterrows():
            p = price_local_vol(spot, o["strike"], T, o["type"], sigma, D, F, surface)
            iv = implied_vol(p, F(T), o["strike"], T, D(T), o["type"])
            rows.append({**o.to_dict(), "lv_price": p, "lv_iv": iv})
    out = pd.DataFrame(rows)
    out["fit_iv"] = surface.implied_vol(out["k"].values, out["T"].values)
    out["err_bp"] = 1e4 * (out["lv_iv"] - out["iv"])            # model vs market
    out["pde_err_bp"] = 1e4 * (out["lv_iv"] - out["fit_iv"])    # numerical error only
    out["in_spread"] = (out["lv_iv"] >= out["iv_bid"]) & (out["lv_iv"] <= out["iv_ask"])
    return out


# =====================================================================
# Plots
# =====================================================================

def plot_all(quotes, surface, sigma, F, rt, spot):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # smiles: solid line over the quoted range, dashed beyond (extrapolation)
    k_all = np.linspace(-0.6, 0.6, 300)
    for T, g in quotes.groupby("T"):
        iv_fit = surface.implied_vol(k_all, T)
        inside = (k_all >= g["k"].min()) & (k_all <= g["k"].max())
        line, = ax[0].plot(k_all[inside], iv_fit[inside], lw=1.2)
        ax[0].plot(k_all, np.where(inside, np.nan, iv_fit), "--", lw=0.8, color=line.get_color())
        ax[0].plot(g["k"], g["iv"], ".", ms=3, color=line.get_color(), label=f"T={T:.2f}")
    ax[0].set(xlabel="ln(K/F)", ylabel="Implied volatility",
              title="Market smiles and SVI fit (dashed: extrapolation)")
    ax[0].legend(fontsize=7)

    S = np.linspace(0.65 * spot, 1.35 * spot, 300)
    for t in np.quantile(surface.T_nodes, [0, 0.33, 0.66, 1]):
        ax[1].plot(S, sigma(S, t), label=f"t={t:.2f}")
    ax[1].axvline(spot, color="grey", lw=0.6, ls=":")
    ax[1].set(xlabel="S", ylabel="Local volatility", title="Local volatility (Gatheral)")
    ax[1].legend(fontsize=7)

    for T, g in rt.groupby("T"):
        ax[2].plot(g["k"], g["err_bp"], "o-", ms=4, label=f"T={T:.2f}")
    ax[2].axhline(0, color="grey", lw=0.8)
    ax[2].set(xlabel="ln(K/F)", ylabel="Local vol − market (vol bp)",
              title="Round trip: implied vol error")
    ax[2].legend(fontsize=7)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Synthetic chain (to test without a connection)
# =====================================================================

def make_synthetic_chain(spot=6800.0, r=0.04, q=0.013, seed=0):
    """
    SPX-like chain: strong put skew, few far-OTM calls at short maturities, bid/ask and
    noise. The true surface is an arbitrage-free SSVI surface, written as raw SVI slices.
    """
    rng = np.random.default_rng(seed)
    T_nodes = np.array([0.04, 0.06, 0.08, 0.13, 0.35, 0.6, 0.9, 1.6])
    atm_vol = np.array([0.14, 0.142, 0.145, 0.15, 0.16, 0.165, 0.17, 0.175])
    rho, eta, gamma = -0.75, 1.1, 0.45

    true_params = []
    for T, v in zip(T_nodes, atm_vol):
        th = v**2 * T
        phi = eta / (th**gamma * (1 + th) ** (1 - gamma))
        # SSVI slice -> raw SVI parameters
        true_params.append([0.5 * th * (1 - rho**2), 0.5 * th * phi, rho, -rho / phi,
                            np.sqrt(1 - rho**2) / phi])
    truth = SVISurface([SVISlice(T, p, -np.inf, np.inf) for T, p in zip(T_nodes, true_params)])

    rows = []
    for T in T_nodes:
        F, D = spot * np.exp((r - q) * T), np.exp(-r * T)
        sd = 0.15 * np.sqrt(T)
        k_lo, k_hi = max(-0.6, -9 * sd), min(0.5, 3 * sd)   # few far-OTM calls at short T
        for K in np.unique(np.round(F * np.exp(np.linspace(k_lo, k_hi, 120)) / 5) * 5):
            iv = truth.implied_vol(np.log(K / F), T)
            for opt_type in ("call", "put"):
                mid = black_price(F, K, T, D, iv, opt_type) * (1 + 0.003 * rng.standard_normal())
                half = max(0.05, 0.01 * mid)
                rows.append({"T": T, "strike": K, "type": opt_type,
                             "bid": round(mid - half, 2), "ask": round(mid + half, 2)})
    return spot, pd.DataFrame(rows), truth


# =====================================================================
# Run
# =====================================================================

def run(spot, raw, plot=True):
    df = clean_chain(raw)
    curve = forward_and_discount(df, spot)
    print("Implied curve (put-call parity):")
    print(curve.round(4).to_string(index=False))
    if curve["replaced"].any():
        print("  [!] noisy or implausible rate replaced by the smooth discount curve at T =",
              curve.loc[curve["replaced"], "T"].round(3).tolist())

    D, F = make_curves(curve, spot)
    quotes = otm_quotes(df, curve)
    surface, quotes = fit_surface(quotes)
    fit_bp = 1e4 * (surface.implied_vol(quotes["k"].values, quotes["T"].values) - quotes["iv"])
    half_spread_bp = 1e4 * 0.5 * (quotes["iv_ask"] - quotes["iv_bid"])
    print(f"\n{len(quotes)} OTM quotes, {len(surface.T_nodes)} maturities")
    print(f"SVI fit RMSE = {np.sqrt(np.mean(fit_bp**2)):.1f} vol bp, "
          f"{(fit_bp.abs() <= half_spread_bp).mean():.0%} of the fitted vols inside the bid-ask spread")
    print("Fit RMSE per maturity (vol bp):",
          {round(T, 2): round(float(np.sqrt(np.mean(e**2))), 1) for T, e in fit_bp.groupby(quotes["T"])})
    check_arbitrage(surface)

    sigma = make_local_vol(surface, F)
    rt = round_trip(quotes, spot, sigma, D, F, surface)
    print(f"\nRound trip vs market: median |error| {rt['err_bp'].abs().median():.1f} vol bp, "
          f"max {rt['err_bp'].abs().max():.1f} vol bp, "
          f"{rt['in_spread'].mean():.0%} of prices inside the bid-ask spread")
    print(f"PDE numerical error (vs SVI fit): median {rt['pde_err_bp'].abs().median():.2f} vol bp, "
          f"max {rt['pde_err_bp'].abs().max():.2f} vol bp")

    if plot:
        plot_all(quotes, surface, sigma, F, rt, spot)
    return curve, quotes, surface, sigma, rt


# =====================================================================
# 9. Comparison with the finite-difference methods of local_vol.py
# =====================================================================
#
#   local_vol.py computes the local vol by Dupire's formula with finite differences on a
#   grid of CALL PRICES, then prices with an implicit scheme on a uniform grid in S.
#   Here, the price grid is generated from the fitted SVI surface (market data are only
#   available at quoted strikes), and both methods are compared on the market quotes:
#     local vol:  Dupire FD on prices      vs  Gatheral (analytic, from the SVI fit)
#     PDE:        uniform S grid, implicit vs  ln S grid, Crank-Nicolson

def d1(f, h, axis):
    """Second-order first derivative along `axis` (uniform grid with step h)."""
    f = np.moveaxis(f, axis, 0)
    out = np.empty_like(f)
    out[1:-1] = (f[2:] - f[:-2]) / (2 * h)                  # central
    out[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * h)        # forward
    out[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / (2 * h)     # backward
    return np.moveaxis(out, 0, axis)


def d2(f, h, axis):
    """Second-order second derivative along `axis` (uniform grid with step h)."""
    f = np.moveaxis(f, axis, 0)
    out = np.empty_like(f)
    out[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / h**2                    # central
    out[0] = (2 * f[0] - 5 * f[1] + 4 * f[2] - f[3]) / h**2              # forward
    out[-1] = (2 * f[-1] - 5 * f[-2] + 4 * f[-3] - f[-4]) / h**2         # backward
    return np.moveaxis(out, 0, axis)


def dupire_local_vol(C, K_grid, T_grid, r, q, tol=1e-12):
    """
    Dupire local volatility surface by finite differences on call prices.
    C: array (len(K_grid), len(T_grid)). r and q can be scalars or arrays of
    instantaneous rates at each maturity, shape (len(T_grid),).
    Returns an array of the same shape as C, with NaN where the formula is not usable.
    """
    hK, hT = K_grid[1] - K_grid[0], T_grid[1] - T_grid[0]
    r = np.broadcast_to(np.asarray(r, float), T_grid.shape)[None, :]
    q = np.broadcast_to(np.asarray(q, float), T_grid.shape)[None, :]
    dC_dK = d1(C, hK, axis=0)
    d2C_dK2 = d2(C, hK, axis=0)
    dC_dT = d1(C, hT, axis=1)

    K = K_grid[:, None]                       # broadcast over maturities
    num = dC_dT + (r - q) * K * dC_dK + q * C
    den = 0.5 * K**2 * d2C_dK2

    vol = np.full_like(C, np.nan)
    ok = (den > tol) & (num > 0)
    vol[ok] = np.sqrt(num[ok] / den[ok])
    return vol


def fill_nan_vol(vol, K_grid):
    """Replace NaNs at each maturity by interpolation in K (flat extrapolation)."""
    vol = vol.copy()
    for j in range(vol.shape[1]):
        ok = ~np.isnan(vol[:, j])
        if not ok.any():
            # no valid point at all (e.g. T = 0): copy the next maturity
            vol[:, j] = np.nan
            continue
        vol[~ok, j] = np.interp(K_grid[~ok], K_grid[ok], vol[ok, j])
    # columns that were entirely NaN take the nearest valid column
    valid = np.where(~np.isnan(vol).all(axis=0))[0]
    for j in np.where(np.isnan(vol).all(axis=0))[0]:
        vol[:, j] = vol[:, valid[np.argmin(np.abs(valid - j))]]
    return vol


def make_sigma(vol, K_grid, T_grid):
    """σ(S, t) interpolated on the local vol surface, flat outside the grid."""
    interp = RegularGridInterpolator((K_grid, T_grid), vol)

    def sigma(S, t):
        S = np.clip(S, K_grid[0], K_grid[-1])
        t = np.clip(t, T_grid[0], T_grid[-1])
        return interp(np.column_stack([S, np.full_like(S, t)]))

    return sigma


def fdm_price(S0, K, T, r, q, sigma, S_max=None, M=800, N=400):
    """European call under local volatility, implicit scheme on a uniform grid in S."""
    if S_max is None:
        S_max = 4 * max(S0, K)

    S = np.linspace(0.0, S_max, M + 1)
    dS, dt = S[1] - S[0], T / N
    Si = S[1:-1]                                  # interior nodes

    C = np.maximum(S - K, 0.0)                    # terminal condition C(S, T)

    for n in range(N - 1, -1, -1):                # step backward from T to 0
        t = n * dt
        sig = sigma(Si, t)
        alpha = 0.5 * sig**2 * Si**2 / dS**2
        beta = (r - q) * Si / (2 * dS)
        a, b, c = alpha - beta, -2 * alpha - r, alpha + beta

        C_low = 0.0                               # C(0, t) = 0
        C_up = S_max * np.exp(-q * (T - t)) - K * np.exp(-r * (T - t))   # C(S_max, t)

        rhs = C[1:-1].copy()
        rhs[0] += dt * a[0] * C_low
        rhs[-1] += dt * c[-1] * C_up

        ab = np.zeros((3, M - 1))                 # scipy banded format
        ab[0, 1:] = -dt * c[:-1]                  # upper diagonal
        ab[1, :] = 1 - dt * b                     # main diagonal
        ab[2, :-1] = -dt * a[1:]                  # lower diagonal

        C[1:-1] = solve_banded((1, 1), ab, rhs)
        C[0], C[-1] = C_low, C_up

    return np.interp(S0, S, C), S, C


def price_grid_from_surface(surface, D, F, K_grid, T_grid):
    """Call prices C[i, j] on (K_grid[i], T_grid[j]) from the fitted SVI surface."""
    C = np.empty((len(K_grid), len(T_grid)))
    for j, t in enumerate(T_grid):
        if t == 0:
            C[:, j] = np.maximum(F(0.0) - K_grid, 0.0)
            continue
        Ft, Dt = F(t), D(t)
        iv = np.sqrt(surface.w(np.log(K_grid / Ft), t) / t)
        C[:, j] = black_price(Ft, K_grid, t, Dt, iv, "call")
    return C


def compare_methods(quotes, spot, surface, D, F, per_slice=7, hK_frac=0.005, hT=0.01, plot=True):
    """2 x 2 comparison on the market quotes: {Dupire FD, Gatheral} x {uniform implicit, log CN}."""
    # --- Dupire FD surface on a uniform (K, T) grid ---
    K_grid = np.arange(0.3, 2.0 + 1e-9, hK_frac) * spot
    T_grid = np.arange(0.0, surface.T_nodes[-1] + hT / 2, hT)
    C = price_grid_from_surface(surface, D, F, K_grid, T_grid)
    # instantaneous rates at each maturity: r = -d ln D/dt, q = r - d ln F/dt
    eps = 1e-4
    tt = np.maximum(T_grid, eps)
    r_inst = -(np.log(D(tt + eps)) - np.log(D(tt - eps))) / (2 * eps)
    mu_inst = (np.log(F(tt + eps)) - np.log(F(tt - eps))) / (2 * eps)
    vol_fd = dupire_local_vol(C, K_grid, T_grid, r_inst, r_inst - mu_inst)
    print(f"\nDupire FD grid: {len(K_grid)} strikes x {len(T_grid)} maturities, "
          f"{np.isnan(vol_fd).mean():.1%} NaN before filling")
    sigma_fd = make_sigma(fill_nan_vol(vol_fd, K_grid), K_grid, T_grid)
    sigma_gat = make_local_vol(surface, F)

    # --- pricers; fdm_price takes constant rates: the average ones up to T ---
    def uniform_pde(sigma):
        def pricer(K, T, opt_type):
            r_T = -np.log(D(T)) / T
            q_T = r_T - np.log(F(T) / spot) / T
            c = fdm_price(spot, K, T, r_T, q_T, sigma)[0]
            return c if opt_type == "call" else c - D(T) * (F(T) - K)   # put-call parity
        return pricer

    def log_pde(sigma):
        return lambda K, T, opt_type: price_local_vol(spot, K, T, opt_type, sigma, D, F, surface)

    configs = {
        "Dupire FD + uniform implicit": uniform_pde(sigma_fd),
        "Dupire FD + log CN": log_pde(sigma_fd),
        "Gatheral + uniform implicit": uniform_pde(sigma_gat),
        "Gatheral + log CN": log_pde(sigma_gat),
    }

    sub = []
    for T, g in quotes.groupby("T"):
        g = g.sort_values("k")
        sub.append(g.iloc[np.unique(np.linspace(0, len(g) - 1, per_slice).round().astype(int))])
    sub = pd.concat(sub)

    results = {}
    for name, pricer in configs.items():
        rows = []
        for _, o in sub.iterrows():
            p = pricer(o["strike"], o["T"], o["type"])
            iv = implied_vol(p, F(o["T"]), o["strike"], o["T"], D(o["T"]), o["type"])
            rows.append({"T": o["T"], "k": o["k"], "err_bp": 1e4 * (iv - o["iv"]),
                         "in_spread": o["iv_bid"] <= iv <= o["iv_ask"] if np.isfinite(iv) else False})
        res = pd.DataFrame(rows)
        results[name] = res
        print(f"{name:30s} | median |err| {res['err_bp'].abs().median():6.1f} vol bp | "
              f"max {res['err_bp'].abs().max():7.1f} vol bp | NaN {res['err_bp'].isna().sum():2d} | "
              f"{res['in_spread'].mean():4.0%} inside the bid-ask spread")

    if plot:
        # local vol surfaces at a few maturities
        fig, ax = plt.subplots(figsize=(9, 4.5))
        S = np.linspace(0.6 * spot, 1.4 * spot, 300)
        for t, color in zip(np.quantile(surface.T_nodes, [0, 0.5, 1]), ["C0", "C1", "C2"]):
            ax.plot(S, sigma_fd(S, t), "--", color=color, label=f"Dupire FD, t = {t:.2f}")
            ax.plot(S, sigma_gat(S, t), "-", color=color, label=f"Gatheral, t = {t:.2f}")
        ax.axvline(spot, color="grey", lw=0.6, ls=":")
        ax.set(xlabel="S", ylabel="Local volatility", title="Local volatility: finite differences vs analytic")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

        # round-trip errors, one panel per configuration, same scale
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
        for ax, (name, res) in zip(axes.ravel(), results.items()):
            for T, g in res.groupby("T"):
                ax.plot(g["k"], g["err_bp"], "o-", ms=3, label=f"T={T:.2f}")
            ax.axhline(0, color="grey", lw=0.8)
            ax.set(title=name, xlabel="ln(K/F)", ylabel="Local vol − market (vol bp)")
        axes[0, 0].legend(fontsize=7)
        plt.tight_layout()
        plt.show()
    return results


if __name__ == "__main__":
    USE_MARKET = True
    COMPARE_METHODS = True    # 2 x 2 comparison with the finite-difference methods of local_vol.py
    if USE_MARKET:
        spot, raw = load_chain_yfinance("^SPX", n_expiries=8)
    else:
        spot, raw, _ = make_synthetic_chain()
    curve, quotes, surface, sigma, rt = run(spot, raw)
    if COMPARE_METHODS:
        D, F = make_curves(curve, spot)
        compare_methods(quotes, spot, surface, D, F)
