"""
Homma Signal Pro
Candlestick Trading Bible (Homma) + Risk Management + Fundamentals + Adaptive Learning + Telegram
Educational tool only — NOT financial advice.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
import json
from datetime import datetime
from pathlib import Path

# -------------------------------------------------
# Page Config - Classic clean look
# -------------------------------------------------
st.set_page_config(
    page_title="Homma Signal Pro",
    page_icon="🕯️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Persistence files
# -------------------------------------------------
PERFORMANCE_FILE = Path("pattern_performance.json")
JOURNAL_FILE = Path("trade_journal.csv")

DEFAULT_PERFORMANCE = {
    "Hammer": {"wins": 0, "losses": 0, "breakeven": 0},
    "Shooting Star": {"wins": 0, "losses": 0, "breakeven": 0},
    "Bullish Engulfing": {"wins": 0, "losses": 0, "breakeven": 0},
    "Bearish Engulfing": {"wins": 0, "losses": 0, "breakeven": 0},
    "Morning Star": {"wins": 0, "losses": 0, "breakeven": 0},
    "Evening Star": {"wins": 0, "losses": 0, "breakeven": 0},
    "Doji": {"wins": 0, "losses": 0, "breakeven": 0},
}

INSTRUMENTS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "AUDJPY": "AUDJPY=X",
    "CADJPY": "CADJPY=X",
    "EURAUD": "EURAUD=X",
    "GBPAUD": "GBPAUD=X",
    "GOLD (XAU)": "GC=F",
    "US OIL (WTI)": "CL=F",
    "BRENT OIL": "BZ=F",
    "SILVER": "SI=F",
}

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_performance():
    if PERFORMANCE_FILE.exists():
        try:
            with open(PERFORMANCE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_PERFORMANCE.copy()

def save_performance(data):
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_journal():
    if JOURNAL_FILE.exists():
        try:
            return pd.read_csv(JOURNAL_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "timestamp", "instrument", "pattern", "signal", "entry",
        "stop_loss", "take_profit", "position_size", "risk_amount",
        "outcome", "pnl", "notes"
    ])

def save_journal(df):
    df.to_csv(JOURNAL_FILE, index=False)

def get_pattern_winrate(perf, pattern):
    stats = perf.get(pattern, {"wins": 0, "losses": 0, "breakeven": 0})
    total = stats["wins"] + stats["losses"] + stats["breakeven"]
    if total == 0:
        return None, 0
    return stats["wins"] / total * 100, total

def get_adaptive_weight(perf, pattern):
    winrate, total = get_pattern_winrate(perf, pattern)
    if total < 5:
        return 1.0
    if winrate >= 60: return 1.30
    if winrate >= 50: return 1.15
    if winrate >= 40: return 0.90
    return 0.60

# -------------------------------------------------
# Candlestick Pattern Functions (Homma / Bible basis)
# -------------------------------------------------
def is_doji(o, h, l, c, body_threshold=0.05):
    body = abs(c - o)
    range_ = h - l
    return range_ > 0 and body / range_ < body_threshold

def is_hammer(o, h, l, c, body_ratio=0.3, shadow_ratio=2.0):
    body = abs(c - o)
    range_ = h - l
    if range_ == 0 or body == 0: return False
    lower = min(o, c) - l
    upper = h - max(o, c)
    return body / range_ < body_ratio and lower >= shadow_ratio * body and upper <= body * 0.5

def is_shooting_star(o, h, l, c, body_ratio=0.3, shadow_ratio=2.0):
    body = abs(c - o)
    range_ = h - l
    if range_ == 0 or body == 0: return False
    lower = min(o, c) - l
    upper = h - max(o, c)
    return body / range_ < body_ratio and upper >= shadow_ratio * body and lower <= body * 0.5

def is_bullish_engulfing(prev_o, prev_c, o, c):
    return prev_c < prev_o and c > o and o < prev_c and c > prev_o

def is_bearish_engulfing(prev_o, prev_c, o, c):
    return prev_c > prev_o and c < o and o > prev_c and c < prev_o

def is_morning_star(df, i):
    if i < 2: return False
    c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    return (c1['Close'] < c1['Open'] and
            abs(c1['Close'] - c1['Open']) > (c1['High'] - c1['Low']) * 0.5 and
            abs(c2['Close'] - c2['Open']) < abs(c1['Close'] - c1['Open']) * 0.5 and
            c3['Close'] > c3['Open'] and c3['Close'] > (c1['Open'] + c1['Close']) / 2)

def is_evening_star(df, i):
    if i < 2: return False
    c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    return (c1['Close'] > c1['Open'] and
            abs(c1['Close'] - c1['Open']) > (c1['High'] - c1['Low']) * 0.5 and
            abs(c2['Close'] - c2['Open']) < abs(c1['Close'] - c1['Open']) * 0.5 and
            c3['Close'] < c3['Open'] and c3['Close'] < (c1['Open'] + c1['Close']) / 2)

def detect_patterns(df, performance):
    signals = []
    n = len(df)
    if n < 5: return signals

    for i in range(max(2, n-12), n):
        row = df.iloc[i]
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        date = row.name.strftime('%Y-%m-%d') if hasattr(row.name, 'strftime') else str(row.name)
        candidates = []

        if is_doji(o, h, l, c):
            candidates.append(("Doji", "Neutral", "WAIT", "Medium"))
        if is_hammer(o, h, l, c):
            recent_down = c < df['Close'].iloc[max(0, i-5)]
            candidates.append(("Hammer", "Bullish Reversal", "BUY" if recent_down else "WATCH", "Strong" if recent_down else "Medium"))
        if is_shooting_star(o, h, l, c):
            recent_up = c > df['Close'].iloc[max(0, i-5)]
            candidates.append(("Shooting Star", "Bearish Reversal", "SELL" if recent_up else "WATCH", "Strong" if recent_up else "Medium"))

        if i >= 1:
            prev = df.iloc[i-1]
            if is_bullish_engulfing(prev['Open'], prev['Close'], o, c):
                candidates.append(("Bullish Engulfing", "Bullish Reversal", "BUY", "Strong"))
            if is_bearish_engulfing(prev['Open'], prev['Close'], o, c):
                candidates.append(("Bearish Engulfing", "Bearish Reversal", "SELL", "Strong"))

        if is_morning_star(df, i):
            candidates.append(("Morning Star", "Bullish Reversal", "BUY", "Very Strong"))
        if is_evening_star(df, i):
            candidates.append(("Evening Star", "Bearish Reversal", "SELL", "Very Strong"))

        for pattern, ptype, signal, strength in candidates:
            weight = get_adaptive_weight(performance, pattern)
            winrate, total = get_pattern_winrate(performance, pattern)
            note = f"Weight {weight:.2f}x"
            if total > 0:
                note += f" | Win rate {winrate:.1f}% ({total} trades)"
            signals.append({
                "date": date, "pattern": pattern, "type": ptype,
                "signal": signal, "strength": strength, "weight": weight, "note": note
            })
    return signals[-10:] if signals else []

def get_overall_signal(signals, fundamental_bias):
    if not signals:
        return "HOLD", "No clear candlestick pattern.", "gray", None, 0

    buy_score = sum(s["weight"] for s in signals if s["signal"] == "BUY")
    sell_score = sum(s["weight"] for s in signals if s["signal"] == "SELL")
    strong_buy = any(s["signal"] == "BUY" and "Strong" in s["strength"] for s in signals[-4:])
    strong_sell = any(s["signal"] == "SELL" and "Strong" in s["strength"] for s in signals[-4:])
    best = max(signals, key=lambda x: x["weight"] if x["signal"] in ("BUY", "SELL") else 0)

    # Fundamental alignment bonus/penalty
    alignment = 0
    if fundamental_bias == "Bullish":
        buy_score *= 1.25
        sell_score *= 0.75
        alignment = 1
    elif fundamental_bias == "Bearish":
        sell_score *= 1.25
        buy_score *= 0.75
        alignment = -1

    if strong_buy and buy_score >= sell_score:
        conf = "High" if alignment == 1 else "Medium"
        return "BUY", f"Strong bullish candlestick + fundamental bias: {fundamental_bias}", "green", best, alignment
    if strong_sell and sell_score >= buy_score:
        conf = "High" if alignment == -1 else "Medium"
        return "SELL", f"Strong bearish candlestick + fundamental bias: {fundamental_bias}", "red", best, alignment
    if buy_score > sell_score * 1.15:
        return "LEAN BUY", "More bullish evidence (candlestick + fundamentals)", "lightgreen", best, alignment
    if sell_score > buy_score * 1.15:
        return "LEAN SELL", "More bearish evidence (candlestick + fundamentals)", "salmon", best, alignment
    return "HOLD", "Mixed signals or low confidence. Wait.", "orange", best, alignment

# -------------------------------------------------
# Risk Management
# -------------------------------------------------
def calculate_atr(df, period=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def risk_management(df, direction, account_size, risk_percent, rr_ratio=2.0, atr_mult=1.5):
    if df is None or len(df) < 20: return None
    entry = float(df["Close"].iloc[-1])
    atr = calculate_atr(df)
    if pd.isna(atr) or atr <= 0: atr = entry * 0.01

    if direction == "BUY":
        stop_loss = entry - atr * atr_mult
        risk_per_unit = entry - stop_loss
        take_profit = entry + risk_per_unit * rr_ratio
    else:
        stop_loss = entry + atr * atr_mult
        risk_per_unit = stop_loss - entry
        take_profit = entry - risk_per_unit * rr_ratio

    risk_amount = account_size * (risk_percent / 100)
    if risk_per_unit <= 0: return None
    position_size = risk_amount / risk_per_unit

    return {
        "entry": round(entry, 5),
        "stop_loss": round(stop_loss, 5),
        "take_profit": round(take_profit, 5),
        "atr": round(atr, 5),
        "risk_per_unit": round(risk_per_unit, 5),
        "position_size": round(position_size, 2),
        "risk_amount": round(risk_amount, 2),
        "rr_ratio": rr_ratio
    }

# -------------------------------------------------
# Telegram (supports Secrets + sidebar)
# -------------------------------------------------
def get_telegram_credentials(sidebar_token, sidebar_chat):
    # Prefer Streamlit Secrets if available
    try:
        token = st.secrets["telegram"]["token"]
        chat_id = st.secrets["telegram"]["chat_id"]
        return token, str(chat_id)
    except Exception:
        return sidebar_token, sidebar_chat

def send_telegram(token, chat_id, message):
    if not token or not chat_id:
        return False, "Token or Chat ID missing"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }, timeout=10)
        if r.status_code == 200:
            return True, "Sent successfully"
        return False, r.text
    except Exception as e:
        return False, str(e)

# -------------------------------------------------
# Data
# -------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_data(ticker, period="6mo", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        data.columns = [str(c).capitalize() for c in data.columns]
        data = data.dropna()
        needed = ["Open", "High", "Low", "Close"]
        if not all(c in data.columns for c in needed): return None
        return data[needed]
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

# -------------------------------------------------
# UI - Classic & Organized
# -------------------------------------------------
st.title("🕯️ Homma Signal Pro")
st.caption("Candlestick Trading Bible (Munehisa Homma)  •  Risk Management  •  Fundamentals  •  Adaptive Learning")

st.warning("⚠️ Educational tool only. Not financial advice. Past results do not guarantee future performance.")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("1. Account & Risk")
    account_size = st.number_input("Account Size ($)", min_value=100.0, value=10000.0, step=100.0)
    risk_percent = st.slider("Risk per trade (%)", 0.25, 2.0, 0.5, 0.25)
    rr_ratio = st.selectbox("Risk : Reward Ratio", [1.5, 2.0, 2.5, 3.0], index=1)
    atr_mult = st.slider("ATR Stop Multiplier", 1.0, 3.0, 1.5, 0.25)

    st.subheader("2. Fundamental Bias")
    fundamental_bias = st.radio(
        "Current Fundamental View",
        ["Neutral", "Bullish", "Bearish"],
        index=0,
        help="Aligns with interest rates, economic data, risk sentiment etc."
    )
    high_impact_news = st.checkbox("High-impact news expected soon?", value=False)

    st.subheader("3. Telegram Alerts")
    st.caption("Leave empty if using Streamlit Secrets")
    tg_token = st.text_input("Bot Token", type="password", placeholder="Paste token here")
    tg_chat = st.text_input("Chat ID", placeholder="Paste chat id here")
    enable_tg = st.checkbox("Send alert on strong BUY/SELL", value=True)

    st.markdown("---")
    st.caption("Adaptive learning updates when you log trades below.")

# ===== MAIN CONTROLS =====
st.subheader("Market Selection")
c1, c2, c3 = st.columns(3)
with c1:
    instrument = st.selectbox("Instrument", list(INSTRUMENTS.keys()))
with c2:
    period = st.selectbox("Lookback Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
with c3:
    interval = st.selectbox("Timeframe", ["1d", "4h", "1h"], index=0)

ticker = INSTRUMENTS[instrument]

if st.button("🔍 Scan Market & Generate Full Plan", type="primary", use_container_width=True):
    with st.spinner(f"Scanning {instrument} with Homma patterns + fundamentals..."):
        df = fetch_data(ticker, period, interval)
        performance = load_performance()

        if df is None or len(df) < 30:
            st.error("Insufficient data. Try a different instrument or longer period.")
        else:
            signals = detect_patterns(df, performance)
            overall, reason, color, best, alignment = get_overall_signal(signals, fundamental_bias)

            # Color card
            color_map = {
                "green": "#1b5e20", "red": "#b71c1c",
                "lightgreen": "#2e7d32", "salmon": "#c62828",
                "orange": "#ef6c00", "gray": "#455a64"
            }
            bg = color_map.get(color, "#455a64")

            st.markdown("---")
            st.markdown(f"""
            <div style="background:{bg};padding:28px;border-radius:10px;text-align:center;color:white;margin-bottom:20px;">
                <h1 style="margin:0;font-size:2.8rem;letter-spacing:1px;">{overall}</h1>
                <p style="margin:8px 0 0;font-size:1.15rem;opacity:0.95;">{reason}</p>
            </div>
            """, unsafe_allow_html=True)

            if high_impact_news:
                st.warning("⚠️ High-impact news flagged — consider reducing size or waiting.")

            # Risk Plan
            st.subheader("🛡️ Risk Management Plan")
            direction = "BUY" if "BUY" in overall else ("SELL" if "SELL" in overall else None)
            risk_plan = None

            if direction:
                risk_plan = risk_management(df, direction, account_size, risk_percent, rr_ratio, atr_mult)
                if risk_plan:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Entry", risk_plan["entry"])
                    m2.metric("Stop Loss", risk_plan["stop_loss"])
                    m3.metric("Take Profit", risk_plan["take_profit"])
                    m4.metric("Position Size", risk_plan["position_size"])
                    st.info(f"Risk Amount: **${risk_plan['risk_amount']}**  |  ATR: {risk_plan['atr']}  |  R:R = 1:{risk_plan['rr_ratio']}")
                else:
                    st.warning("Could not calculate risk parameters.")
            else:
                st.info("No clear directional signal → no position calculated.")

            # Chart
            st.subheader("Price Chart")
            fig = go.Figure(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name=instrument
            ))
            if risk_plan:
                fig.add_hline(y=risk_plan["entry"], line_dash="dot", line_color="white", annotation_text="Entry")
                fig.add_hline(y=risk_plan["stop_loss"], line_dash="dash", line_color="#ff5252", annotation_text="SL")
                fig.add_hline(y=risk_plan["take_profit"], line_dash="dash", line_color="#69f0ae", annotation_text="TP")
            fig.update_layout(
                title=f"{instrument} • {interval}",
                xaxis_rangeslider_visible=False,
                height=460,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Patterns
            st.subheader("Detected Candlestick Patterns")
            if signals:
                for s in reversed(signals):
                    emoji = "🟢" if s["signal"] == "BUY" else ("🔴" if s["signal"] == "SELL" else "⚪")
                    with st.expander(f"{emoji}  {s['date']}  —  {s['pattern']}  →  {s['signal']}  (weight {s['weight']:.2f}x)"):
                        st.write(f"**Type:** {s['type']}  |  **Strength:** {s['strength']}")
                        st.write(s["note"])
            else:
                st.info("No strong classic patterns in the recent candles.")

            # Telegram
            if enable_tg and direction and risk_plan and overall in ("BUY", "SELL"):
                token, chat_id = get_telegram_credentials(tg_token, tg_chat)
                msg = f"""
🕯️ <b>Homma Signal Pro</b>

Instrument: <b>{instrument}</b>
Signal: <b>{overall}</b>
Pattern: {best['pattern'] if best else '—'}
Fundamental Bias: {fundamental_bias}

Entry: <code>{risk_plan['entry']}</code>
Stop Loss: <code>{risk_plan['stop_loss']}</code>
Take Profit: <code>{risk_plan['take_profit']}</code>
Position Size: <code>{risk_plan['position_size']}</code>
Risk: ${risk_plan['risk_amount']}

{datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
                ok, info = send_telegram(token, chat_id, msg.strip())
                if ok:
                    st.success("✅ Telegram notification sent successfully!")
                else:
                    st.error(f"Telegram error: {info}")

# ===== JOURNAL & LEARNING =====
st.markdown("---")
st.header("📓 Trade Journal & Adaptive Learning")

with st.expander("➕ Log a completed trade (updates the learning system)"):
    with st.form("journal_form"):
        jc1, jc2 = st.columns(2)
        with jc1:
            log_inst = st.selectbox("Instrument", list(INSTRUMENTS.keys()))
            log_pattern = st.selectbox("Pattern", list(DEFAULT_PERFORMANCE.keys()))
            log_signal = st.selectbox("Direction", ["BUY", "SELL"])
        with jc2:
            log_outcome = st.selectbox("Outcome", ["Win", "Loss", "Breakeven"])
            log_pnl = st.number_input("P&L ($)", value=0.0, step=5.0)
            log_notes = st.text_input("Notes")
        if st.form_submit_button("Save & Update Learning"):
            perf = load_performance()
            key = log_outcome.lower()
            if key == "win":
                perf[log_pattern]["wins"] += 1
            elif key == "loss":
                perf[log_pattern]["losses"] += 1
            else:
                perf[log_pattern]["breakeven"] += 1
            save_performance(perf)

            journal = load_journal()
            new_row = {
                "timestamp": datetime.now().isoformat(),
                "instrument": log_inst, "pattern": log_pattern, "signal": log_signal,
                "entry": "", "stop_loss": "", "take_profit": "",
                "position_size": "", "risk_amount": "",
                "outcome": log_outcome, "pnl": log_pnl, "notes": log_notes
            }
            journal = pd.concat([journal, pd.DataFrame([new_row])], ignore_index=True)
            save_journal(journal)
            st.success(f"Logged. {log_pattern} statistics updated.")
            st.rerun()

st.subheader("Pattern Performance (Adaptive Weights)")
perf = load_performance()
rows = []
for p, s in perf.items():
    total = s["wins"] + s["losses"] + s["breakeven"]
    wr = round(s["wins"] / total * 100, 1) if total else "—"
    rows.append({"Pattern": p, "Wins": s["wins"], "Losses": s["losses"], "BE": s["breakeven"], "Total": total, "Win Rate %": wr})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

journal = load_journal()
if not journal.empty:
    st.subheader("Recent Journal")
    st.dataframe(journal.tail(12), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Core logic based on classic Japanese candlestick principles originating with Munehisa Homma. Data: Yahoo Finance. This is an educational prototype.")
