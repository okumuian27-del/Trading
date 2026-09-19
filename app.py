"""
Forex Candlestick Signal App
Inspired by Munehisa Homma's candlestick principles
as popularized in "The Candlestick Trading Bible".
Educational tool only — NOT financial advice.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Candlestick Signal App | Homma Style",
    page_icon="🕯️",
    layout="wide"
)

# --------------------------
# Pattern Detection Functions
# (Standard definitions aligned with Homma / Candlestick Trading Bible style)
# --------------------------

def is_doji(o, h, l, c, body_threshold=0.05):
    """Doji: open ≈ close, indecision. Tight threshold for cleaner signals."""
    body = abs(c - o)
    range_ = h - l
    if range_ == 0:
        return False
    return body / range_ < body_threshold

def is_hammer(o, h, l, c, body_ratio=0.3, shadow_ratio=2.0):
    """
    Hammer (bullish reversal): small body near top, long lower shadow.
    Typically after downtrend.
    """
    body = abs(c - o)
    range_ = h - l
    if range_ == 0 or body == 0:
        return False
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    return (body / range_ < body_ratio and
            lower_shadow >= shadow_ratio * body and
            upper_shadow <= body * 0.5)

def is_inverted_hammer(o, h, l, c, body_ratio=0.3, shadow_ratio=2.0):
    """Inverted Hammer (potential bullish): small body near bottom, long upper shadow."""
    body = abs(c - o)
    range_ = h - l
    if range_ == 0 or body == 0:
        return False
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    return (body / range_ < body_ratio and
            upper_shadow >= shadow_ratio * body and
            lower_shadow <= body * 0.5)

def is_shooting_star(o, h, l, c, body_ratio=0.3, shadow_ratio=2.0):
    """Shooting Star (bearish reversal): small body near bottom, long upper shadow after uptrend."""
    return is_inverted_hammer(o, h, l, c, body_ratio, shadow_ratio)

def is_bullish_engulfing(prev_o, prev_h, prev_l, prev_c, o, h, l, c):
    """Bullish Engulfing: current green candle fully engulfs previous red candle."""
    prev_bearish = prev_c < prev_o
    curr_bullish = c > o
    engulfs = o < prev_c and c > prev_o
    return prev_bearish and curr_bullish and engulfs

def is_bearish_engulfing(prev_o, prev_h, prev_l, prev_c, o, h, l, c):
    """Bearish Engulfing: current red candle fully engulfs previous green candle."""
    prev_bullish = prev_c > prev_o
    curr_bearish = c < o
    engulfs = o > prev_c and c < prev_o
    return prev_bullish and curr_bearish and engulfs

def is_morning_star(df, i):
    """Morning Star (bullish reversal 3-candle): long red, small body (gap down), long green."""
    if i < 2:
        return False
    c1 = df.iloc[i-2]
    c2 = df.iloc[i-1]
    c3 = df.iloc[i]
    # First candle strong bearish
    first_bearish = c1['Close'] < c1['Open'] and abs(c1['Close'] - c1['Open']) > (c1['High'] - c1['Low']) * 0.5
    # Second small body (doji-like or spinning top)
    second_small = abs(c2['Close'] - c2['Open']) < abs(c1['Close'] - c1['Open']) * 0.5
    # Third strong bullish closing into first body
    third_bullish = c3['Close'] > c3['Open'] and c3['Close'] > (c1['Open'] + c1['Close']) / 2
    return first_bearish and second_small and third_bullish

def is_evening_star(df, i):
    """Evening Star (bearish reversal 3-candle)."""
    if i < 2:
        return False
    c1 = df.iloc[i-2]
    c2 = df.iloc[i-1]
    c3 = df.iloc[i]
    first_bullish = c1['Close'] > c1['Open'] and abs(c1['Close'] - c1['Open']) > (c1['High'] - c1['Low']) * 0.5
    second_small = abs(c2['Close'] - c2['Open']) < abs(c1['Close'] - c1['Open']) * 0.5
    third_bearish = c3['Close'] < c3['Open'] and c3['Close'] < (c1['Open'] + c1['Close']) / 2
    return first_bullish and second_small and third_bearish

def detect_patterns(df):
    """Scan the last few candles for patterns and return signals."""
    signals = []
    n = len(df)
    if n < 5:
        return signals

    # Simple trend filter: SMA 20 vs SMA 50 or recent closes
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean() if n >= 50 else df['Close'].rolling(min(20, n)).mean()

    # Look at last 5 candles for recent patterns
    for i in range(max(2, n-10), n):
        row = df.iloc[i]
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        date = row.name.strftime('%Y-%m-%d') if hasattr(row.name, 'strftime') else str(row.name)

        # Single candle
        if is_doji(o, h, l, c):
            signals.append({
                'date': date,
                'pattern': 'Doji',
                'type': 'Neutral / Indecision',
                'signal': 'WAIT',
                'strength': 'Medium',
                'note': 'Market indecision. Wait for confirmation. Classic Homma psychology signal.'
            })
        if is_hammer(o, h, l, c):
            # Prefer after down move
            recent_down = df['Close'].iloc[i] < df['Close'].iloc[max(0, i-5)]
            sig = 'BUY' if recent_down else 'WATCH'
            signals.append({
                'date': date,
                'pattern': 'Hammer',
                'type': 'Bullish Reversal',
                'signal': sig,
                'strength': 'Strong' if recent_down else 'Medium',
                'note': 'Buyers stepped in after selling pressure. Strong if after decline (Homma / Bible style).'
            })
        if is_shooting_star(o, h, l, c):
            recent_up = df['Close'].iloc[i] > df['Close'].iloc[max(0, i-5)]
            sig = 'SELL' if recent_up else 'WATCH'
            signals.append({
                'date': date,
                'pattern': 'Shooting Star',
                'type': 'Bearish Reversal',
                'signal': sig,
                'strength': 'Strong' if recent_up else 'Medium',
                'note': 'Sellers rejected higher prices. Strong after an advance.'
            })

        # Two-candle
        if i >= 1:
            prev = df.iloc[i-1]
            if is_bullish_engulfing(prev['Open'], prev['High'], prev['Low'], prev['Close'], o, h, l, c):
                signals.append({
                    'date': date,
                    'pattern': 'Bullish Engulfing',
                    'type': 'Bullish Reversal',
                    'signal': 'BUY',
                    'strength': 'Strong',
                    'note': 'Buyers completely overwhelmed sellers. High-probability reversal signal from the Candlestick Bible.'
                })
            if is_bearish_engulfing(prev['Open'], prev['High'], prev['Low'], prev['Close'], o, h, l, c):
                signals.append({
                    'date': date,
                    'pattern': 'Bearish Engulfing',
                    'type': 'Bearish Reversal',
                    'signal': 'SELL',
                    'strength': 'Strong',
                    'note': 'Sellers completely overwhelmed buyers. Classic bearish reversal.'
                })

        # Three-candle
        if is_morning_star(df, i):
            signals.append({
                'date': date,
                'pattern': 'Morning Star',
                'type': 'Bullish Reversal',
                'signal': 'BUY',
                'strength': 'Very Strong',
                'note': 'Three-candle bullish reversal. Powerful Homma-style bottoming pattern.'
            })
        if is_evening_star(df, i):
            signals.append({
                'date': date,
                'pattern': 'Evening Star',
                'type': 'Bearish Reversal',
                'signal': 'SELL',
                'strength': 'Very Strong',
                'note': 'Three-candle bearish reversal. Powerful topping pattern.'
            })

    # Deduplicate by keeping the most recent strong ones
    return signals[-8:] if signals else []


def get_overall_signal(signals):
    """Aggregate recent signals into one recommendation."""
    if not signals:
        return 'HOLD / NO CLEAR SIGNAL', 'No strong candlestick pattern detected recently. Wait for clearer price action.', 'gray'

    # Prioritize strongest most recent
    buy_count = sum(1 for s in signals if s['signal'] == 'BUY')
    sell_count = sum(1 for s in signals if s['signal'] == 'SELL')
    strong_buy = any(s['signal'] == 'BUY' and 'Strong' in s['strength'] for s in signals[-3:])
    strong_sell = any(s['signal'] == 'SELL' and 'Strong' in s['strength'] for s in signals[-3:])

    if strong_buy and buy_count >= sell_count:
        return 'BUY', 'Recent strong bullish candlestick pattern(s) detected (Homma / Candlestick Bible principles).', 'green'
    elif strong_sell and sell_count >= buy_count:
        return 'SELL', 'Recent strong bearish candlestick pattern(s) detected.', 'red'
    elif buy_count > sell_count:
        return 'LEAN BUY', 'More bullish patterns than bearish recently. Confirm with trend.', 'lightgreen'
    elif sell_count > buy_count:
        return 'LEAN SELL', 'More bearish patterns than bullish recently.', 'salmon'
    else:
        return 'HOLD / WAIT', 'Mixed or indecisive signals (Doji etc.). Patience is key per Homma psychology.', 'orange'


# --------------------------
# Data Fetching
# --------------------------

@st.cache_data(ttl=3600)
def fetch_forex_data(pair: str, period: str = "6mo", interval: str = "1d"):
    """Fetch OHLC data using yfinance. Pair like EURUSD=X"""
    ticker = f"{pair}=X"
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty:
            return None
        # Flatten multi-index columns if present (yfinance often returns MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        # Normalize column names to title case
        data.columns = [str(c).capitalize() for c in data.columns]
        data = data.dropna()
        needed = ['Open', 'High', 'Low', 'Close']
        available = [c for c in needed if c in data.columns]
        if len(available) < 4:
            return None
        return data[needed]
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


# --------------------------
# UI
# --------------------------

st.title("🕯️ Forex Candlestick Trading Signals")
st.markdown("""
**Inspired by Munehisa Homma** (the father of Japanese candlesticks) and the principles popularized in  
*"The Candlestick Trading Bible"*.

This app detects classic candlestick patterns and generates educational **BUY / SELL / WAIT** signals.
""")

st.warning("⚠️ **Educational purposes only.** This is NOT financial advice. Past patterns do not guarantee future results. Always do your own research and risk management.")

col1, col2, col3 = st.columns(3)

with col1:
    pair = st.selectbox(
        "Currency Pair",
        options=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"],
        index=0
    )

with col2:
    period = st.selectbox(
        "Lookback Period",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=2
    )

with col3:
    interval = st.selectbox(
        "Timeframe",
        options=["1d", "4h", "1h"],
        index=0,
        help="Daily is best for classic Homma-style patterns"
    )

if st.button("🔍 Generate Signals", type="primary"):
    with st.spinner(f"Fetching {pair} data and scanning candlestick patterns..."):
        df = fetch_forex_data(pair, period, interval)

        if df is None or len(df) < 20:
            st.error("Could not fetch enough data. Try another pair or period.")
        else:
            signals = detect_patterns(df)
            overall, reason, color = get_overall_signal(signals)

            # Overall Signal Card
            st.markdown("---")
            st.subheader(f"📊 Current Recommendation for {pair}")

            signal_color_map = {
                'green': '#00c853',
                'red': '#d50000',
                'lightgreen': '#69f0ae',
                'salmon': '#ff8a80',
                'orange': '#ffab00',
                'gray': '#9e9e9e'
            }
            bg = signal_color_map.get(color, '#9e9e9e')

            st.markdown(f"""
            <div style="background-color:{bg}; padding: 25px; border-radius: 12px; text-align:center; color:white;">
                <h1 style="margin:0; font-size: 2.8em;">{overall}</h1>
                <p style="margin:8px 0 0; font-size:1.1em;">{reason}</p>
            </div>
            """, unsafe_allow_html=True)

            # Chart
            st.subheader("Candlestick Chart")
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=pair
            )])
            fig.update_layout(
                title=f"{pair} — {interval} Candles",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                height=500,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detected Patterns
            st.subheader("Detected Patterns (Recent)")
            if signals:
                for s in reversed(signals):
                    emoji = "🟢" if s['signal'] == 'BUY' else ("🔴" if s['signal'] == 'SELL' else "⚪")
                    with st.expander(f"{emoji} {s['date']} — {s['pattern']} ({s['type']}) → {s['signal']}"):
                        st.write(f"**Strength:** {s['strength']}")
                        st.write(f"**Note:** {s['note']}")
            else:
                st.info("No clear classic patterns detected in the most recent candles. Market may be ranging or trending without strong reversal signals.")

            # Educational section
            with st.expander("📖 Quick Reference — Key Patterns from Homma / Candlestick Bible Style"):
                st.markdown("""
                | Pattern | Bias | Meaning (Psychology) |
                |---------|------|----------------------|
                | **Hammer** | Bullish | Sellers pushed price down but buyers rejected the lows — potential bottom |
                | **Shooting Star** | Bearish | Buyers pushed high but sellers rejected — potential top |
                | **Bullish Engulfing** | Bullish | Current green candle completely swallows previous red — buyers take control |
                | **Bearish Engulfing** | Bearish | Current red candle completely swallows previous green — sellers take control |
                | **Morning Star** | Strong Bullish | 3-candle bottom: long red → small body → long green |
                | **Evening Star** | Strong Bearish | 3-candle top: long green → small body → long red |
                | **Doji** | Neutral | Open ≈ Close → pure indecision. Wait for next candle confirmation |

                **Homma's core insight:** Markets are driven by emotion (fear & greed). Candlesticks visualize that battle.
                Always confirm with higher-timeframe trend and risk management.
                """)

st.markdown("---")
st.caption("Built as an educational prototype. Data via Yahoo Finance. Patterns follow widely known Japanese candlestick definitions originating with Munehisa Homma.")
