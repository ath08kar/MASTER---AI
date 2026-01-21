import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# Technical Analysis Library
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# PDF Generation Libraries
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ============================================================
# 🧠 IMPORT CUSTOM AGENTS
# ============================================================
try:
    from agents.market_researcher import market_research
    from agents.orchestrator import orchestrator_decision
except ImportError:
    st.error("⚠️ Critical Error: Could not import your 'agents'. Please ensure the 'agents' folder exists.")
    st.stop()

# ============================================================
# 🎨 UI CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="MASTER.AI — Alpha Insight Engine",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling - Dynamic Theme Injection handled below after Sidebar configuration


# ============================================================
# ⚙️ SYSTEM SETTINGS & CONSTANTS
# ============================================================
DIRECTORIES = {
    "reports": "reports",
    "images": "temp_images",
    "models": "models"
}
for d in DIRECTORIES.values():
    os.makedirs(d, exist_ok=True)

# Map UI selection to yfinance periods and roughly how many days to slice for charts
TIMEFRAME_MAP = {
    "1 Day":   {"period": "1d", "days": 1},
    "1 Week":  {"period": "5d", "days": 5},
    "1 Month": {"period": "1mo", "days": 30},
    "3 Months": {"period": "3mo", "days": 90},
    "6 Months": {"period": "6mo", "days": 180},
    "1 Year":  {"period": "1y", "days": 365}
}

# ============================================================
# 🛠️ DATA & MODEL ENGINE
# ============================================================

class MarketEngine:
    
    @staticmethod
    @st.cache_resource
    def load_brain():
        model_path = os.path.join(DIRECTORIES["models"], "technical_analyst.pkl")
        try:
            bundle = joblib.load(model_path)
            return bundle["model"], bundle["features"]
        except:
            return None, None

    @staticmethod
    def fetch_market_data(ticker, timeframe_config):
        """
        Smart Fetch: 
        The ML model ALWAYS needs ~200+ days of data for indicators (EMA/RSI).
        Even if the user selects "1 Week", we fetch 1 Year of data for the backend,
        but we will slice it for the UI later.
        """
        try:
            # Always fetch at least 1y for robust indicator calculation
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df.dropna()
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_stock_info(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            def get_val(key, default="N/A"): return info.get(key, default)
            return {
                "name": get_val("longName", ticker),
                "sector": get_val("sector", "Unknown"),
                "mcap": get_val("marketCap", 0)
            }
        except:
            return {"name": ticker, "sector": "-", "mcap": 0}

    @staticmethod
    def engineer_features(df):
        df = df.copy()
        df["rsi"] = RSIIndicator(df["Close"], 14).rsi()
        df["ema_fast"] = EMAIndicator(df["Close"], 12).ema_indicator()
        df["ema_slow"] = EMAIndicator(df["Close"], 26).ema_indicator()
        df["ema_cross"] = (df["ema_fast"] > df["ema_slow"]).astype(int)
        df["atr"] = AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()
        df["ret"] = df["Close"].pct_change()
        df["vol_20"] = df["ret"].rolling(20).std()
        return df

    @staticmethod
    def predict_next_day(df):
        """Simple Linear Regression on last 30 days to project next day price."""
        if len(df) < 30: return 0.0
        
        subset = df.tail(30).reset_index()
        y = subset["Close"].values.reshape(-1, 1)
        X = np.array(subset.index).reshape(-1, 1)
        
        reg = LinearRegression().fit(X, y)
        next_index = np.array([[30]]) # The next integer index
        pred_price = reg.predict(next_index)[0][0]
        return pred_price

# ============================================================
# 📝 PDF REPORT GENERATOR
# ============================================================

class PDFGenerator:
    def generate(self, filename, ticker_info, analysis_data, chart_path, logo_path=None):
        doc = SimpleDocTemplate(filename, pagesize=LETTER)
        styles = getSampleStyleSheet()
        elements = []
        
        # --- PAGE 1: TITLE & LOGO ---
        if logo_path and os.path.exists(logo_path):
            try:
                # Resize logo while maintaining aspect ratio if needed, or just specific width
                img = RLImage(logo_path, width=200, height=200, kind='proportional')
                elements.append(img)
            except: pass
            
        elements.append(Spacer(1, 60))
        
        main_title = ParagraphStyle('MainTitle', parent=styles['Heading1'], fontSize=32, textColor=colors.HexColor("#1e3a8a"), alignment=1, spaceAfter=20)
        elements.append(Paragraph("MASTER.AI", main_title))
        
        sub_title = ParagraphStyle('SubTitle', parent=styles['Heading2'], fontSize=18, textColor=colors.HexColor("#4b5563"), alignment=1)
        elements.append(Paragraph("Intelligence Report", sub_title))
        
        elements.append(Spacer(1, 100))
        elements.append(Paragraph(f"Created for: {ticker_info['name']}", styles['Title']))
        elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
        
        elements.append(PageBreak())
        
        # --- PAGE 2: VERDICT & CHART ---
        elements.append(Paragraph(f"Analysis Snapshot: {ticker_info['name']}", styles['Heading2']))
        elements.append(Spacer(1, 20))
        
        # Verdict Box
        decision = analysis_data['decision']['decision']
        color = colors.green if decision == "BUY" else colors.red if decision == "SELL" else colors.grey
        t = Table([[f"AI VERDICT: {decision}"]], colWidths=[400], rowHeights=[50])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), color), 
            ('TEXTCOLOR', (0,0), (-1,-1), colors.white), 
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), 
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), 
            ('FONTSIZE', (0,0), (-1,-1), 20),
            ('ROUNDEDCORNERS', [10, 10, 10, 10])
        ]))
        elements.append(t)
        
        elements.append(Spacer(1, 25))
        elements.append(Paragraph(f"<b>Prediction:</b> Projected Price: {analysis_data['next_day_pred']:.2f}", styles['Normal']))
        elements.append(Paragraph(f"<b>Rationale:</b> {analysis_data['decision']['reason']}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        if os.path.exists(chart_path):
            try:
                img = RLImage(chart_path, width=450, height=250)
                elements.append(img)
            except: pass
            
        elements.append(PageBreak())
        
        # --- PAGE 3: DETAILED INTELLIGENCE ---
        elements.append(Paragraph("Market Intelligence & Details", styles['Heading2']))
        elements.append(Spacer(1, 20))
        
        # Stock Info Table
        data = [
            ["Sector", ticker_info['sector']],
            ["Market Cap", f"{ticker_info['mcap']:,}"],
            ["Current Regime", analysis_data['market_out']['regime']],
            ["Volatility", analysis_data['market_out']['volatility']]
        ]
        t_info = Table(data, colWidths=[150, 300])
        t_info.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
            ('PADDING', (0,0), (-1,-1), 8)
        ]))
        elements.append(t_info)
        
        elements.append(Spacer(1, 30))
        elements.append(Paragraph("<b>Market Commentary:</b>", styles['Heading3']))
        
        # Parse explanation safely
        explanation_text = analysis_data['market_out'].get('explanation', 'No detailed commentary available.')
        elements.append(Paragraph(explanation_text, styles['Normal']))
        
        elements.append(Spacer(1, 20))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
        elements.append(Paragraph("<i>Disclaimer: This report is generated by AI agents for informational purposes only. It does not constitute financial advice.</i>", disclaimer_style))
            
        doc.build(elements)
        return filename

# ============================================================
# 🖥️ MAIN APPLICATION LOGIC
# ============================================================

def main():
    # 1. Initialize Session State
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "data" not in st.session_state:
        st.session_state.data = {}

    tech_model, model_features = MarketEngine.load_brain()
    
    # ---------------- Sidebar ----------------
    with st.sidebar:
        # Placement of Logo
        logo_path = "WhatsApp_Image_2025-12-22_at_11.52.56_AM-removebg-preview - Copy.png"
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
            
        st.header("🎛️ Control Panel")
        
        # Theme Toggle
        dark_mode = st.toggle("🌙 Dark Mode", value=True)
        
        ticker_input = st.text_input("Stock Symbol", "RELIANCE").upper()
        exchange = st.radio("Exchange", ["NSE", "BSE"], horizontal=True)
        
        st.markdown("---")
        st.markdown("##### ⏳ Time Horizon")
        
        # New Timeframe Selector
        time_selection = st.selectbox(
            "Select View Period",
            list(TIMEFRAME_MAP.keys()),
            index=5 # Default to 1 Year
        )
        
        confidence_threshold = st.slider("AI Confidence Threshold", 0.50, 0.90, 0.60, 0.05)
        
        st.markdown("---")
        
        # Only run analysis if button clicked OR if we haven't run it yet
        if st.button("⚡ Ignite Analysis", type="primary", use_container_width=True):
            st.session_state.analysis_done = False # Reset to force fresh run
            st.session_state.trigger_run = True # Signal to run below

    # Define Colors based on theme
    if dark_mode: # Dark Mode
        bg_color = "#0e1117"
        text_color = "#fafafa"
        card_bg = "#262730"
    else: # Light Mode (Clean White/Gray)
        bg_color = "#ffffff"
        text_color = "#000000" 
        card_bg = "#f0f2f6"

    # Inject CSS
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        .verdict-box {{
            font-size: 2.2rem;
            font-weight: 800;
            text-align: center;
            padding: 15px;
            border-radius: 12px;
            margin: 20px 0;
            color: #ffffff; /* Always white text for verdict signal box */
        }}
        .buy-signal {{ background: linear-gradient(135deg, #064e3b, #065f46); color: #a7f3d0; border: 1px solid #059669; }}
        .sell-signal {{ background: linear-gradient(135deg, #7f1d1d, #991b1b); color: #fecaca; border: 1px solid #dc2626; }}
        .wait-signal {{ background: linear-gradient(135deg, #1f2937, #111827); color: #e5e7eb; border: 1px solid #4b5563; }}
        
        /* Metrics Styling */
        div[data-testid="stMetricValue"] {{ font-size: 1.4rem !important; }}
    </style>
    """, unsafe_allow_html=True)

    # ---------------- Execution Logic ----------------
    # Check if triggered by button
    if st.session_state.get("trigger_run", False):
        ticker = f"{ticker_input}.{'NS' if exchange == 'NSE' else 'BO'}"
        
        with st.status("Neural Network Active...", expanded=True) as status:
            st.write("Uplinking Market Data...")
            # Fetch long history for ML, slice later for view
            df_full = MarketEngine.fetch_market_data(ticker, TIMEFRAME_MAP[time_selection])
            
            if df_full is None or len(df_full) < 120:
                status.update(label="🚫 Data Access Denied", state="error")
                st.error("Not enough historical data.")
                st.stop()
                
            info = MarketEngine.get_stock_info(ticker)
            
            st.write("📐 Calculating Vectors...")
            df_features = MarketEngine.engineer_features(df_full)
            
            # Predict Next Day
            next_day_price = MarketEngine.predict_next_day(df_full)

            # ML Inference
            st.write(" Decoding Patterns (ML)...")
            if tech_model:
                latest_features = df_features.iloc[-1:][model_features].apply(pd.to_numeric, errors="coerce").dropna()
                probs = tech_model.predict_proba(latest_features)[0]
            else:
                probs = [0.33, 0.33, 0.33] # Fallback if no model

            st.write("🌐 Scanning Global Sentiment...")
            ticker_obj = yf.Ticker(ticker)
            news_items = ticker_obj.news[:10]  # Fetch top 10 news items
            headlines = [n.get("title", "") for n in news_items]
            
            market_out = market_research(df_full, headlines or [f"{ticker} trend", f"{ticker} news"])
            
            st.write("⚖️ Weighing Probability...")
            decision = orchestrator_decision(
                market_out["regime"], 
                market_out["confidence"], 
                probs, 
                confidence_threshold
            )
            
            # SAVE TO SESSION STATE
            st.session_state.data = {
                "ticker": ticker,
                "df": df_full,
                "info": info,
                "market_out": market_out,
                "probs": probs,
                "decision": decision,
                "next_day": next_day_price,
                "news": news_items
            }
            st.session_state.analysis_done = True
            st.session_state.trigger_run = False # Turn off trigger
            status.update(label="✨ Insight Generated", state="complete", expanded=False)

    # ---------------- Dashboard Rendering ----------------
    # Only render if analysis is stored in session state
    if st.session_state.analysis_done:
        data = st.session_state.data
        df = data["df"]
        
        # --- Slice Data based on Timeframe Selection ---
        days_to_show = TIMEFRAME_MAP[time_selection]["days"]
        if len(df) > days_to_show:
            df_view = df.tail(days_to_show)
        else:
            df_view = df
        
        # Header
        st.markdown(f"### Market Report: {data['info']['name']}")
        
        curr_price = float(df["Close"].iloc[-1])
        prev_price = float(df["Close"].iloc[-2])
        pct_change = ((curr_price - prev_price) / prev_price) * 100
        
        # Metrics with Prediction
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"₹{curr_price:,.2f}", f"{pct_change:.2f}%")
        c2.metric("🎯 Price Target", f"₹{data['next_day']:,.2f}")
        c3.metric("Market Regime", data['market_out']["regime"])
        c4.metric("AI Confidence", f"{float(max(data['probs']))*100:.1f}%")
        
        st.markdown("---")
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader(f"Price Action ({time_selection})")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=df_view.index, open=df_view['Open'], high=df_view['High'], low=df_view['Low'], close=df_view['Close'], name='OHLC'), row=1, col=1)
            fig.add_trace(go.Bar(x=df_view.index, y=df_view['Volume'], name='Volume', marker_color='rgba(100, 100, 255, 0.3)'), row=2, col=1)
            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Save chart for PDF
            chart_filename = os.path.join(DIRECTORIES["images"], "chart_temp.png")
            try: fig.write_image(chart_filename)
            except: pass 

        with col_right:
            st.subheader("AI Verdict")
            d_text = data["decision"]["decision"]
            css_class = "wait-signal"
            if d_text == "BUY": css_class = "buy-signal"
            elif d_text == "SELL": css_class = "sell-signal"
            
            st.markdown(f"""<div class="verdict-box {css_class}">{d_text}</div>""", unsafe_allow_html=True)
            st.info(data["decision"]["reason"])
            
            st.markdown("#### Model Probabilities")
            probs = data["probs"]
            st.caption(f"Bullish: {float(probs[2]):.2%}")
            st.progress(float(probs[2]))
            st.caption(f"Bearish: {float(probs[0]):.2%}")
            st.progress(float(probs[0]))

        # --- Live News Feed ---
        st.markdown("---")
        st.subheader("📰 Live News Feed")
        if data.get("news"):
            for item in data["news"]:
                with st.container():
                    col_news_1, col_news_2 = st.columns([0.8, 0.2])
                    with col_news_1:
                        st.markdown(f"**[{item.get('title')}]({item.get('link')})**")
                        st.caption(f"Source: {item.get('publisher')} | Type: {item.get('type')}")
                    with col_news_2:
                        # Extract thumbnail if available
                        thumbnail = item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url')
                        if thumbnail:
                            st.image(thumbnail, width=100)
                    st.markdown("---")
        else:
            st.write("No recent news found for this ticker.")

        # PDF Generation - NOW WORKS WITHOUT RELOADING
        st.markdown("---")
        if st.button("Generate PDF Report"):
            with st.spinner("Compiling PDF..."):
                pdf_gen = PDFGenerator()
                snapshot = {
                    "decision": data["decision"],
                    "next_day_pred": data["next_day"],
                    "market_out": data["market_out"]
                }
                report_path = os.path.join(DIRECTORIES["reports"], f"{data['ticker']}_Report.pdf")
                # Define logo path - assuming it's in the root
                logo_path = "WhatsApp_Image_2025-12-22_at_11.52.56_AM-removebg-preview - Copy.png"
                
                pdf_gen.generate(report_path, data["info"], snapshot, chart_filename, logo_path)
                
                with open(report_path, "rb") as f:
                    st.download_button("💾 Save Report", f, file_name=f"{data['ticker']}_Report.pdf", mime="application/pdf")
    else:
        st.markdown("""<div style='text-align: center; padding: 50px;'><h1>  Welcome to Master.AI</h1><p>Enter a symbol and click <b>Ignite Analysis</b>.</p></div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        print("\n" + "="*50)
        print("⚠️  WARNING: You are running this script with 'python'.")
        print("   Please use: streamlit run app.py")
        print("="*50 + "\n")