import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Pong AI", layout="centered")

st.title("🏓 Ping Pong AI Training Visualization")

if os.path.exists("Pong.gif"):
    st.image("Pong.gif", use_column_width=True)

df = pd.read_csv("training_log_v2.csv")

df["SmoothedReward"] = df["TotalReward"].rolling(window=100).mean()
fig1, ax1 = plt.subplots()
ax1.plot(df["Episode"], df["SmoothedReward"], color="blue", linewidth=1)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_title("Total Reward (Smoothed)")
ax1.grid(True)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(df["Episode"], df["Epsilon"], color="green", linewidth=1)
ax2.set_xlabel("Episode")
ax2.set_ylabel("Epsilon")
ax2.set_title("Epsilon Decay")
ax2.grid(True)
st.pyplot(fig2)

st.dataframe(df.tail(20))
st.dataframe(df.head(10))
