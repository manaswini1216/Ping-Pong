import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ping Pong AI Training", layout="centered")

# Title and Description
st.title("🏓 Ping Pong AI — Q-Learning Training Dashboard")
st.markdown("""
This dashboard demonstrates how a reinforcement learning agent was trained to play Ping Pong using **Q-learning**.
""")

# Load training data
st.header("📊 Training Progress")
try:
    df = pd.read_csv("training_log.csv")
    st.success("Loaded training_log.csv successfully!")

    # Plot total reward
    st.subheader("Total Reward per Episode")
    fig1, ax1 = plt.subplots()
    ax1.plot(df["Episode"], df["TotalReward"], color="teal")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Reward Curve")
    st.pyplot(fig1)

    # Plot epsilon decay
    st.subheader("Epsilon Decay (Exploration Rate)")
    fig2, ax2 = plt.subplots()
    ax2.plot(df["Episode"], df["Epsilon"], color="orange")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Epsilon Decay")
    st.pyplot(fig2)

except FileNotFoundError:
    st.error("❌ training_log.csv not found. Please run `train.py` to generate it.")

# Optional: Show a GIF
st.header("🎮 AI Paddle Gameplay Demo")
try:
    st.image("pong.gif", caption="AI vs Human Gameplay", use_column_width=True)
except:
    st.info("Add a 'pong.gif' file to show gameplay here.")

# How it works section
st.header("🧠 How It Works")
st.markdown("""
- The left paddle is controlled by a Q-learning agent.
- It observes the game state: **ball position**, **ball velocity**, and **paddle position**.
- It chooses from 3 actions: stay, move up, or move down.
- Over 20,000 episodes, the agent learns using:
    - **Positive reward** (+1) for hitting the ball  
    - **Negative reward** (-1) for missing  
    - **Small penalty** (-0.01) each frame to reduce jittering
- A Q-table is updated using the Bellman equation to guide future decisions.
""")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit, Pygame, and Q-learning")
