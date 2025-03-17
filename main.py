import streamlit as st
import datetime as dt
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from pandas_datareader import data as pdr
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#######################
# Page configuration
st.set_page_config(
    page_title="Black Scholes Pricing Asset",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

# CSS
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
    text-align: center;
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
    text-align: center;
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
    text-align: centre;
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
    text-align: center;
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)

ttm_options = ['A Week', 'A Month', 'A Year']

# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("Created by:")
    linkedin_url = "https://www.linkedin.com/in/prytm/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Priya Tammam`</a>', unsafe_allow_html=True)

    st.header("Input Data")
    stocks = st.text_input("Underlying Asset", value = "AAPL")
    strike = st.number_input("Strike Price", value= 200.00)
    time_to_maturity = st.selectbox("Time To Maturity", options = ttm_options, index=ttm_options.index('A Week'))
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

    st.markdown("---")

    with st.expander("📘 What is an Option?"):
        st.markdown("""
        **Options** are financial derivatives that give the buyer the **right**, but not the obligation, to **buy (Call)** or **sell (Put)** an underlying asset at a specified price (strike price) before or at a certain date.
        
        - **Call Option** → Gives the right to **buy** the asset → used when you expect the price to **go up**.
        - **Put Option** → Gives the right to **sell** the asset → used when you expect the price to **go down**.
        
        Options can be used for:
        - 📈 **Speculation**: Predicting price movement without owning the asset.
        - 🛡️ **Hedging**: Protecting your portfolio from adverse price movements.
        
        > Options are powerful tools in finance — they’re not just for speculation, but also risk management!
        
        ---
        **Example**  
        Suppose AAPL stock is at **\$213**. You buy a Call Option with a strike price of **\$200**. If AAPL rises to **\$230**, you can buy it for **\$200** — earning a **\$30** profit (minus premium).
        """)
    
    with st.expander("**📘 Whats is Black-Scholes Model?**"):
        st.markdown("""
        The **Black-Scholes Model** is a mathematical model used to calculate the theoretical price of European-style options (both Call and Put).
    
        Developed by Fischer Black, Myron Scholes, and Robert Merton in 1973, this model assumes that asset prices follow a log-normal distribution and incorporates several key variables:
        - Current asset price
        - Strike price
        - Time to maturity
        - Asset price volatility
        - Risk-free interest rate
    
        The Black-Scholes model is widely used in financial markets because it provides a relatively simple yet powerful method to estimate option prices.
    
        **Note:** The model assumes no dividends and does not apply to American options, which can be exercised before the expiration date.
        """)
        
    with st.expander("📊 What are Greeks in Options?"):
        st.markdown("""
        **Greeks** are risk measures that show how an option’s price is expected to change with various factors:
        
        - **Delta (Δ)**: Measures how much the option price changes when the **underlying asset** price changes by $1.  
            > High Delta = Option is more sensitive to price changes.
        
        - **Gamma (Γ)**: Measures how fast Delta changes when the asset price changes.  
            > Think of Gamma as the "acceleration" of Delta.
        
        - **Theta (Θ)**: Measures the **time decay** of the option.  
            > Shows how much value the option loses each day as it approaches expiry.
        
        - **Vega (ν)**: Measures how much the option price changes with a **1% change in volatility** of the asset.  
            > More volatility = higher option value.
        
        - **Rho (ρ)**: Measures sensitivity to **interest rate** changes.  
            > Least impactful in most short-term options.
        
        ---
        These help traders and analysts manage risk and build more advanced strategies.
        """)

# Time to Maturity
if time_to_maturity == 'A Week':
    ttm = 1/52
elif time_to_maturity == 'A Month':
    ttm = 1/12
else:
    ttm = 1

# (Include the BlackScholes class definition here)
def BlackScholes (S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return call_price, put_price

# Stocks df
end = dt.datetime.now()
start = end - dt.timedelta(days=120)

df = yf.download(stocks, start, end)
df.columns = df.columns.get_level_values(0)

# Stocks Volatility
log_return = np.log(df['Close'] / df['Close'].shift(1))
log_return = log_return.dropna()
volatility = log_return.rolling(window=30).std() * np.sqrt(252)
bs_vol = volatility.iloc[-1]

# Current Price
current_price = df['Close'].iloc[-1]

# Greeks Calculations
def black_scholes_g_c (S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta_c = norm.cdf(d1)
    theta_c = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    rho_c = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    gamma_c = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega_c = S * norm.pdf(d1) * np.sqrt(T) / 100

    return delta_c, theta_c, rho_c, gamma_c, vega_c

delta_c, theta_c, rho_c, gamma_c, vega_c = black_scholes_g_c(current_price, strike, ttm, interest_rate, bs_vol)

def black_scholes_g_p (S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta_p = norm.cdf(d1) - 1
    theta_p = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    rho_p = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma_p = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega_p = S * norm.pdf(d1) * np.sqrt(T) / 100

    return delta_p, theta_p, rho_p, gamma_p, vega_p

delta_p, theta_p, rho_p, gamma_p, vega_p = black_scholes_g_p(current_price, strike, ttm, interest_rate, bs_vol)

# Candle Stick PLot
def plot_candlestick_volume(df, stocks):
    # Calculate Moving Averages & Bollinger Bands
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['BB_upper'] = df['MA10'] + 2 * df['Close'].rolling(window=10).std()
    df['BB_lower'] = df['MA10'] - 2 * df['Close'].rolling(window=10).std()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{stocks} Price Chart', 'Volume'),
        row_width=[0.2, 0.7]
    )

    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ), row=1, col=1)

    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='blue', width=1), name='MA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], line=dict(color='grey', width=1), name='MA200'), row=1, col=1)

    # Bollinger Bands (Dashed Lines)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_upper'],
        line=dict(color='lightgreen', width=1, dash='dash'),
        name='BB Upper'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_lower'],
        line=dict(color='orange', width=1, dash='dash'),
        name='BB Lower'
    ), row=1, col=1)

    # MA10 line
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA10'],
        line=dict(color='purple', width=1),
        name='MA10'
    ), row=1, col=1)

    # Shaded Area Between BB Upper & Lower
    fig.add_trace(go.Scatter(
        x=pd.concat([pd.Series(df.index), pd.Series(df.index[::-1])]),
        y=pd.concat([df['BB_upper'], df['BB_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(173,216,230,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ), row=1, col=1)

    # Volume Bar Chart
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='red', showlegend=False), row=2, col=1)

    # Layout
    fig.update_layout(
        xaxis_tickfont_size=10,
        yaxis=dict(title='Price ($/Share)'),
        autosize=True,
        height=700,
        margin=dict(l=50, r=50, b=100, t=50, pad=4),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update(layout_xaxis_rangeslider_visible=False)
    return fig

# Main Page for Output Display
st.title("Black-Scholes Pricing Model")

# Table of Inputs
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity": [time_to_maturity],
    "Volatility (σ)": [bs_vol],
    "Risk-Free Interest Rate": [interest_rate],
}

input_df = pd.DataFrame(input_data)
st.table(input_df)

put_df = pd.DataFrame(put_data)

# Calculate Call and Put values
call_price, put_price = BlackScholes(current_price, strike, ttm, interest_rate, bs_vol)

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">Optimal CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">Optimal PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")

col3, col4 = st.columns([1,1], gap="medium")

with col3:
    # Data
    call_data = [[delta_c, theta_c, rho_c, gamma_c, vega_c]]
    
    # MultiIndex header
    headers = pd.MultiIndex.from_product([["Call Greeks"], ["Delta", "Theta", "Rho", "Gamma", "Vega"]])
    call_df = pd.DataFrame(call_data, columns=headers)

    # Display using st.table (karena st.dataframe tidak dukung MultiIndex)
    st.dataframe(call_df, hide_index = True)

with col4:
    # Data
    put_data = [[delta_p, theta_p, rho_p, gamma_p, vega_p]]
    
    # MultiIndex header
    headers = pd.MultiIndex.from_product([["Put Greeks"], ["Delta", "Theta", "Rho", "Gamma", "Vega"]])
    call_df = pd.DataFrame(call_data, columns=headers)

    # Display using st.table (karena st.dataframe tidak dukung MultiIndex)
    st.dataframe(call_df, hide_index = True)

st.markdown("")
st.title("Assets Price - Interactive Candlestick Chart")
st.info("Explore how underlying asset prices fluctuate.")

# Interactive Sliders and Heatmaps for Call and Put Options
if not df.empty:
    df['MA50'] = df['Close'].rolling(window = 50, min_periods = 0).mean()
    df['MA200'] = df['Close'].rolling(window = 200, min_periods = 0).mean()

    # Call plot function
    st.subheader(f"{stocks} Candlestick Chart")
    candlestick_fig = plot_candlestick_volume(df, stocks)
    st.plotly_chart(candlestick_fig, use_container_width=True)
else:
    st.warning("⚠️ No data available for the selected ticker and date range.")
