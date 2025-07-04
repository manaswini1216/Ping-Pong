import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Ping Pong AI Training", layout="centered")

st.title("🏓 Ping Pong AI Training Visualization")
st.markdown("Visualize how the Q-learning agent learns to play Pong over time.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Uploaded training log successfully.")
else:
    if os.path.exists("training_log_v2.csv"):
        df = pd.read_csv("training_log_v2.csv")
        st.info("Loaded local training_log_v2.csv")
    else:
        st.error("No training log CSV found. Upload one to continue.")
        st.stop()

st.subheader("📈 Total Reward per Episode")
fig1, ax1 = plt.subplots()
ax1.plot(df["Episode"], df["TotalReward"], color="dodgerblue", linewidth=1)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_title("Learning Progress")
ax1.grid(True)
st.pyplot(fig1)

# --- Epsilon decay ---
st.subheader("📉 Epsilon Decay")
fig2, ax2 = plt.subplots()
ax2.plot(df["Episode"], df["Epsilon"], color="green", linewidth=1)
ax2.set_xlabel("Episode")
ax2.set_ylabel("Epsilon")
ax2.set_title("Exploration vs Exploitation")
ax2.grid(True)
st.pyplot(fig2)

with st.expander("📊 View Raw Training Data (Last 20 Episodes)"):
    st.dataframe(df.tail(20))

# --- Show pong.gif ---
if os.path.exists("Pong.gif"):
    st.subheader("🎞️ Pong AI Gameplay Demo")
    st.image("Pong.gif", caption="AI vs Human — First to 3 Points", use_column_width=True)
else:
    st.info("ℹ️ Pong.gif not found. Run main.py to generate gameplay demo.")

st.markdown("---")
st.caption("Built using Streamlit | Q-learning Pong AI")
