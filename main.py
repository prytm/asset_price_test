import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas
import seaborn as sns

st.set_page_config(page_title="Black-Scholes Model", layout="wide")

# --- Sidebar ---
st.sidebar.markdown("## 📊 Black-Scholes Model")
st.sidebar.markdown("### Created by:")
st.sidebar.markdown(
    "[![LinkedIn](https://img.shields.io/badge/Prudhvi%20Reddy%2C%20Muppala-blue?style=flat&logo=linkedin)](https://www.linkedin.com)"
)

# Inputs
S = st.sidebar.number_input("Current Asset Price", value=100.0, step=1.0, format="%.2f")
K = st.sidebar.number_input("Strike Price", value=100.0, step=1.0, format="%.2f")
T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, step=0.1, format="%.2f")
sigma = st.sidebar.number_input("Volatility", value=0.2, step=0.01, format="%.2f")
r = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.01, format="%.2f")


# --- Black-Scholes Calculation ---
def black_scholes_call_put(S, K, T, sigma, r):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return round(call, 2), round(put, 2)

call_price, put_price = black_scholes_call_put(S, K, T, sigma, r)

# --- Main Content ---
input_data = {
    "Parameter": [
        "Current Asset Price",
        "Strike Price",
        "Time to Maturity (Years)",
        "Volatility (sigma)",
        "Risk-Free Interest Rate"
    ],
    "Value": [S, K, T, sigma, r]
}

df = pd.DataFrame(input_data)

# Tampilkan tabel
st.subheader("Input Parameters Summary")
st.table(df)

col_call, col_put = st.columns(2)
with col_call:
    st.success(f"📈 CALL Value\n\n## ${call_price}")
with col_put:
    st.error(f"📉 PUT Value\n\n## ${put_price}")

# --- Heatmap Section ---
st.markdown("## 🧊 Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

spot_range = np.linspace(50, 150, 11)
vol_range = np.linspace(0.05, 0.5, 11)

call_matrix = np.zeros((len(spot_range), len(vol_range)))
put_matrix = np.zeros((len(spot_range), len(vol_range)))

for i, s in enumerate(spot_range):
    for j, v in enumerate(vol_range):
        c, p = black_scholes_call_put(s, K, T, v, r)
        call_matrix[i, j] = c
        put_matrix[i, j] = p

# --- Plot Heatmaps ---
col_heat1, col_heat2 = st.columns(2)

with col_heat1:
    st.subheader("📊 Call Price Heatmap")
    fig1, ax1 = plt.subplots()
    sns.heatmap(call_matrix, annot=True, fmt=".1f", xticklabels=[f"{v:.2f}" for v in vol_range], yticklabels=[f"{s:.0f}" for s in spot_range], cmap="viridis", ax=ax1)
    ax1.set_xlabel("Volatility (sigma)")
    ax1.set_ylabel("Spot Price")
    st.pyplot(fig1)

with col_heat2:
    st.subheader("📉 Put Price Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(put_matrix, annot=True, fmt=".1f", xticklabels=[f"{v:.2f}" for v in vol_range], yticklabels=[f"{s:.0f}" for s in spot_range], cmap="plasma", ax=ax2)
    ax2.set_xlabel("Volatility (sigma)")
    ax2.set_ylabel("Spot Price")
    st.pyplot(fig2)
