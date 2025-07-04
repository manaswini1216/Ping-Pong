import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ping Pong AI Training", layout="centered")

st.title("🏓 Ping Pong AI Training Visualization")
st.markdown("This dashboard visualizes the training process of a Q-learning agent for Pong.")

uploaded_file = st.file_uploader("📂 Upload training log CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("training_log_v2.csv")
    except FileNotFoundError:
        st.warning("⚠️ No training_log_v2.csv found in this folder.")
        st.stop()

st.subheader("📈 Total Reward per Episode")
fig1, ax1 = plt.subplots()
ax1.plot(df["Episode"], df["TotalReward"], color='blue', linewidth=0.6)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.grid(True)
st.pyplot(fig1)

st.subheader("🔻 Epsilon Decay Over Episodes")
fig2, ax2 = plt.subplots()
ax2.plot(df["Episode"], df["Epsilon"], color='green', linewidth=0.6)
ax2.set_xlabel("Episode")
ax2.set_ylabel("Epsilon")
ax2.grid(True)
st.pyplot(fig2)

with st.expander("📊 View Raw Training Data"):
    st.dataframe(df.tail(20))

if "pong.gif" in df.columns:
    st.image("pong.gif", caption="Gameplay Demo", use_column_width=True)
elif "pong.gif" in os.listdir():
    st.image("pong.gif", caption="Gameplay Demo", use_column_width=True)

st.success("✅ Training visualization complete!")
