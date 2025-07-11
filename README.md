🏓 Ping Pong vs AI – Reinforcement Learning Game
This project is an interactive Ping Pong game where a player competes against an AI agent trained using Reinforcement Learning (RL). The goal is to demonstrate how an RL agent can learn to play a simple game by interacting with the environment.

🧠 Project Overview
The AI agent is trained using Q-learning, a model-free RL algorithm.

The environment is a simple 2D Pong-style game built with Pygame.

The agent learns through trial and error, receiving rewards for hitting the ball and penalties for missing it.

🎮 Game Mechanics
Player controls the left paddle using keyboard (↑ and ↓ arrows).

AI controls the right paddle using the learned Q-table.

The game rewards the AI for hitting the ball and penalizes it for missing.

📈 Training Highlights
Agent trained over 10,000+ episodes.

Rewards and episode scores plotted to visualize learning progress.

Final agent can consistently compete against human input.

🛠️ Technologies Used
Python

Pygame – for game interface

NumPy – for Q-table and state management

Matplotlib – for plotting training results

Reinforcement Learning (Q-Learning)
