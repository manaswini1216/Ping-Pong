import numpy as np
import pickle
import random
import pandas as pd

# Game parameters
WIDTH, HEIGHT = 600, 300
PADDLE_HEIGHT = 50
BALL_SIZE = 10

# Q-learning parameters
NUM_BUCKETS = 12
ACTIONS = [0, 1, 2]  # 0: stay, 1: up, 2: down
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.1

# Q-table
table = np.random.uniform(low=-1, high=1, size=(
    NUM_BUCKETS, NUM_BUCKETS, NUM_BUCKETS, NUM_BUCKETS, NUM_BUCKETS, len(ACTIONS)))

def discretize(value, min_val, max_val):
    ratio = (value - min_val) / (max_val - min_val)
    return min(NUM_BUCKETS - 1, max(0, int(ratio * NUM_BUCKETS)))

def get_state(ball, vx, vy, ai_y):
    return (
        discretize(ball[0], 0, WIDTH),
        discretize(ball[1], 0, HEIGHT),
        discretize(vx, -5, 5),
        discretize(vy, -5, 5),
        discretize(ai_y, 0, HEIGHT - PADDLE_HEIGHT)
    )

# Training
episodes = 20000
training_log = []

for episode in range(episodes):
    ball = [WIDTH // 2, HEIGHT // 2]
    vx = random.choice([-3, -2, -1, 1, 2, 3])
    vy = random.choice([-3, -2, -1, 1, 2, 3])
    ai_y = random.randint(0, HEIGHT - PADDLE_HEIGHT)

    total_reward = 0  # ✅ Initialize reward tracker per episode

    for step in range(1000):
        state = get_state(ball, vx, vy, ai_y)

        if random.random() < EPSILON:
            action = random.choice(ACTIONS)
        else:
            action = np.argmax(table[state])

        if action == 1:
            ai_y -= 5
        elif action == 2:
            ai_y += 5
        ai_y = max(0, min(ai_y, HEIGHT - PADDLE_HEIGHT))

        # Ball movement
        ball[0] += vx
        ball[1] += vy

        # Bounce off walls + add variation
        if ball[1] <= 0 or ball[1] >= HEIGHT - BALL_SIZE:
            vy *= -1
            vy += random.choice([-1, 0, 1])
            vy = max(-5, min(5, vy))

        reward = -0.01  # default frame penalty

        # Check for paddle hit
        if ball[0] <= 20:
            if ai_y < ball[1] < ai_y + PADDLE_HEIGHT:
                reward = 1
                vx *= -1
            else:
                reward = -1
                break  # game over

        total_reward += reward  # ✅ Track reward

        # Q-value update
        new_state = get_state(ball, vx, vy, ai_y)
        max_future_q = np.max(table[new_state])
        current_q = table[state + (action,)]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        table[state + (action,)] = new_q

    # Log episode stats
    training_log.append((episode, total_reward, EPSILON))

    # Epsilon decay
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

    if episode % 1000 == 0:
        print(f"Episode {episode}/{episodes}")

# Save Q-table
print("✅ Training complete. Saving table_20000.pkl...")
with open("table_20000.pkl", "wb") as f:
    pickle.dump(table, f)
print("📦 Saved as table_20000.pkl")

# Save training log
df = pd.DataFrame(training_log, columns=["Episode", "TotalReward", "Epsilon"])
df.to_csv("training_log.csv", index=False)
print("📄 Saved training_log.csv")
