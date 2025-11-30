import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from textblob import TextBlob # For Sentiment Analysis

# --- Configuration ---
st.set_page_config(
    page_title="ProStock | Professional Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional UI (Light Mode) ---
st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #F0F2F6;
        border: 1px solid #D1D5DB;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #000000;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
        background-color: #000000; 
        color: #FFFFFF;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E5E7EB;
    }
    
    /* Analysis Box */
    .ai-analysis {
        background-color: #e8f4f8;
        border-left: 5px solid #0066cc;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        color: #0f172a;
    }

    /* Hide Streamlit default menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("📈 ProStock Analysis")
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.sidebar.markdown("---")

ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()

# Timeframe Selection for "Live" feel
timeframe = st.sidebar.selectbox(
    "Chart Timeframe",
    ["1 Minute", "5 Minute", "1 Hour", "1 Day"],
    index=0
)

# Map UI selection to yfinance interval/period
if timeframe == "1 Minute":
    interval = "1m"
    period = "1d" # 1m data is best viewed over 1 day
elif timeframe == "5 Minute":
    interval = "5m"
    period = "5d"
elif timeframe == "1 Hour":
    interval = "1h"
    period = "1mo"
else:
    interval = "1d"
    period = "1y" # Default daily view

# Only show Date Input if we are in Daily mode (History mode)
if interval == "1d":
    start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
else:
    st.sidebar.caption(f"⚡ Live Mode: Showing last {period} of {interval} data.")

st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("SMA (Simple Moving Average)", value=True)
sma_period = st.sidebar.number_input("SMA Period", value=20)
show_ema = st.sidebar.checkbox("EMA (Exp. Moving Average)")
ema_period = st.sidebar.number_input("EMA Period", value=50)
show_bb = st.sidebar.checkbox("Bollinger Bands")
show_rsi = st.sidebar.checkbox("RSI (Relative Strength Index)")

# --- Helper Functions ---
@st.cache_data(ttl=60) # Cache data for 60 seconds
def get_stock_data(ticker, interval, period, start=None, end=None):
    try:
        if interval == "1d" and start and end:
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        else:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data(ttl=300)
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info, stock.news
    except Exception as e:
        return None, None

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- Fear & Greed Proxy ---
def get_fear_and_greed_proxy():
    """
    Simulates a Fear & Greed index using VIX and Market Momentum (S&P500).
    Real F&G index requires scraping which is unstable.
    """
    try:
        # Fetch VIX (Volatility)
        vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
        
        # Fetch S&P 500 Momentum (Price vs 125-day avg)
        sp500 = yf.Ticker("^GSPC").history(period="6mo")
        current_sp = sp500['Close'].iloc[-1]
        avg_sp = sp500['Close'].mean()
        
        # Calculate Fear Component (VIX)
        # VIX usually 10-30. >30 is Fear, <20 is Greed.
        # Normalize VIX to 0-100 scale (inverted: high VIX = low score/Fear)
        fear_score = max(0, min(100, 100 - (vix - 10) * 2.5))
        
        # Calculate Momentum Component
        # Above avg = Greed, Below avg = Fear
        momentum_score = max(0, min(100, 50 + ((current_sp - avg_sp) / avg_sp) * 500))
        
        # Composite Score
        final_score = (fear_score * 0.4) + (momentum_score * 0.6)
        
        # Determine Label
        if final_score < 25: label = "Extreme Fear"
        elif final_score < 45: label = "Fear"
        elif final_score < 55: label = "Neutral"
        elif final_score < 75: label = "Greed"
        else: label = "Extreme Greed"
        
        return int(final_score), label
    except:
        return 50, "Neutral" # Fallback

def analyze_news_sentiment(news_items):
    """Analyzes sentiment of news headlines using TextBlob."""
    if not news_items:
        return 0, 0, 0, "Neutral"
    
    polarities = []
    for item in news_items:
        title = item.get('title')
        if not title and 'content' in item:
            title = item['content'].get('title')
            
        if title:
            blob = TextBlob(title)
            polarities.append(blob.sentiment.polarity)
            
    if not polarities:
        return 0, 0, 0, "Neutral"
        
    avg_polarity = np.mean(polarities)
    
    # Count breakdown
    pos = sum(1 for p in polarities if p > 0.1)
    neg = sum(1 for p in polarities if p < -0.1)
    neu = len(polarities) - pos - neg
    
    if avg_polarity > 0.1: sentiment_label = "Positive"
    elif avg_polarity < -0.1: sentiment_label = "Negative"
    else: sentiment_label = "Neutral"
    
    return pos, neg, neu, sentiment_label

def generate_ai_commentary(ticker, current_price, sma, rsi, pct_change, fg_score, fg_label, news_sentiment):
    """Generates a comprehensive AI analysis report."""
    commentary = f"**🤖 AI Market Analyst Report for {ticker}**\n\n"
    
    # 1. Market Sentiment (Fear & Greed)
    commentary += f"**1. Market Sentiment (Fear & Greed):**\n"
    commentary += f"The broader market is currently in a state of **{fg_label} ({fg_score}/100)**. "
    if fg_score < 40:
        commentary += "Investors are fearful, which can sometimes present a buying opportunity for strong assets (contrarian view).\n\n"
    elif fg_score > 60:
        commentary += "Investors are greedy. Caution is advised as the market might be overextended.\n\n"
    else:
        commentary += "Market sentiment is balanced, waiting for a clear direction.\n\n"

    # 2. News Sentiment
    commentary += f"**2. News Analysis:**\n"
    commentary += f"Recent news headlines for {ticker} show a **{news_sentiment}** sentiment. "
    if news_sentiment == "Positive":
        commentary += "Media coverage is optimistic, which often supports price growth.\n\n"
    elif news_sentiment == "Negative":
        commentary += "Media coverage is pessimistic. Watch for potential bad news reactions.\n\n"
    else:
        commentary += "News flow is neutral or mixed, having limited immediate impact on price.\n\n"

    # 3. Technical Analysis
    commentary += f"**3. Technical Signals:**\n"
    trend = "Bullish" if current_price > sma else "Bearish"
    commentary += f"• **Trend:** {trend} (Price vs SMA)\n"
    
    if rsi > 70:
        rsi_signal = "Overbought (High risk of pullback)"
    elif rsi < 30:
        rsi_signal = "Oversold (Potential bounce candidate)"
    else:
        rsi_signal = "Neutral"
    commentary += f"• **Momentum (RSI):** {rsi:.0f} - {rsi_signal}\n"
    
    return commentary

# --- Main App Logic ---

# Fetch Data
if ticker:
    # Handle optional date args for Daily mode
    s_date = start_date if interval == "1d" else None
    e_date = end_date if interval == "1d" else None
    
    data = get_stock_data(ticker, interval, period, s_date, e_date)
    info, news = get_stock_info(ticker)

    if data is not None and len(data) > 0:
        # Clean data structure
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Calculate Indicators
        data['SMA'] = data['Close'].rolling(window=sma_period).mean()
        data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
        data['RSI'] = calculate_rsi(data)
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()

        # --- Dashboard Header ---
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = data['Close'].iloc[-1]
        
        if len(data) > 1:
            prev_close = data['Close'].iloc[-2]
            price_change = current_price - prev_close
            pct_change = (price_change / prev_close) * 100
        else:
            price_change = 0
            pct_change = 0
        
        market_cap = info.get('marketCap', 'N/A')
        volume = info.get('volume', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')

        with col1:
            st.metric(label=f"Price ({interval})", value=f"${current_price:.2f}", delta=f"{price_change:.2f} ({pct_change:.2f}%)")
        with col2:
            st.metric(label="Market Cap", value=f"{market_cap:,.0f}" if isinstance(market_cap, (int, float)) else market_cap)
        with col3:
            st.metric(label="Volume", value=f"{volume:,.0f}" if isinstance(volume, (int, float)) else volume)
        with col4:
            st.metric(label="P/E Ratio", value=f"{pe_ratio}")

        # --- Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart", "🧠 AI Analysis", "📰 Live News", "📋 Info"])

        # Tab 1: Chart
        with tab1:
            st.subheader(f"{ticker} {interval} Chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                name='Price', increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
            ))
            if show_sma: fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], line=dict(color='#FFA500', width=1), name=f'SMA {sma_period}'))
            if show_ema: fig.add_trace(go.Scatter(x=data.index, y=data['EMA'], line=dict(color='#00CED1', width=1), name=f'EMA {ema_period}'))
            if show_bb:
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='#808080', width=1, dash='dash'), name='Upper BB'))
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='#808080', width=1, dash='dash'), name='Lower BB'))
            fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: AI Analysis
        with tab2:
            st.subheader("🧠 Deep Dive AI Analysis")
            
            # 1. Fetch Fear & Greed Data
            fg_score, fg_label = get_fear_and_greed_proxy()
            
            # 2. Analyze News Sentiment
            pos_news, neg_news, neu_news, news_label = analyze_news_sentiment(news)
            
            col_fg, col_news = st.columns(2)
            
            with col_fg:
                st.markdown("### 😨 Fear & Greed Index")
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = fg_score,
                    title = {'text': f"Market Sentiment: {fg_label}"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "black"},
                        'steps': [
                            {'range': [0, 25], 'color': "#FF4136"}, # Extreme Fear
                            {'range': [25, 45], 'color': "#FF851B"}, # Fear
                            {'range': [45, 55], 'color': "#FFDC00"}, # Neutral
                            {'range': [55, 75], 'color': "#2ECC40"}, # Greed
                            {'range': [75, 100], 'color': "#01FF70"} # Extreme Greed
                        ]
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.caption("*Based on Market Volatility (VIX) & Momentum*")

            with col_news:
                st.markdown("### 📰 News Sentiment Analysis")
                st.metric("Overall News Sentiment", news_label)
                
                # Sentiment Bar Chart
                sentiment_data = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Count': [pos_news, neu_news, neg_news]
                })
                fig_sent = go.Figure(go.Bar(
                    x=sentiment_data['Sentiment'],
                    y=sentiment_data['Count'],
                    marker_color=['#2ECC40', '#FFDC00', '#FF4136']
                ))
                fig_sent.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), title="Headline Sentiment Breakdown")
                st.plotly_chart(fig_sent, use_container_width=True)

            st.markdown("---")
            
            # Generate Text Report
            if len(data) > sma_period:
                latest_sma = data['SMA'].iloc[-1]
                latest_rsi = data['RSI'].iloc[-1]
                ai_report = generate_ai_commentary(ticker, current_price, latest_sma, latest_rsi, pct_change, fg_score, fg_label, news_label)
                
                st.markdown(f"""
                <div class="ai-analysis">
                    {ai_report.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Not enough data for full report.")

        # Tab 3: Live News
        with tab3:
            st.subheader(f"Latest News for {ticker}")
            if news:
                for item in news[:10]:
                    title = item.get('title')
                    if not title and 'content' in item: title = item['content'].get('title')
                    if not title: continue 

                    link = item.get('link')
                    if not link: link = item.get('url')
                    if not link and 'clickThroughUrl' in item: 
                        if isinstance(item['clickThroughUrl'], dict): link = item['clickThroughUrl'].get('url')
                    if not link and 'content' in item: link = item['content'].get('clickThroughUrl', {}).get('url')
                    if not link: link = f"https://finance.yahoo.com/quote/{ticker}/news"
                    
                    publisher = item.get('publisher', 'Yahoo Finance')
                    try:
                        publish_time = item.get('providerPublishTime')
                        time_str = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M') if publish_time else "Recent"
                    except: time_str = "Recent"

                    # Analyze individual sentiment
                    blob = TextBlob(title)
                    pol = blob.sentiment.polarity
                    if pol > 0.1: sent_color = "#2ECC40" # Green
                    elif pol < -0.1: sent_color = "#FF4136" # Red
                    else: sent_color = "#999999" # Grey

                    st.markdown(f"""
                    <div style='background-color: #F0F2F6; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #E5E7EB; border-left: 5px solid {sent_color};'>
                        <a href="{link}" target="_blank" style="text-decoration: none; color: #0066CC; font-size: 16px; font-weight: bold;">{title}</a>
                        <p style='color: #333333; font-size: 12px; margin-top: 5px;'>Publisher: {publisher} | {time_str}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No news found.")

        # Tab 4: Info
        with tab4:
            st.subheader("Company Fundamentals")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Country:** {info.get('country', 'N/A')}")
            with f_col2:
                st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
                st.write(f"**Website:** {info.get('website', 'N/A')}")
            st.markdown("### Business Summary")
            st.write(info.get('longBusinessSummary', 'N/A'))

    else:
        st.error("Invalid Ticker or No Data Found.")
else:
    st.info("Enter a ticker to start.")
