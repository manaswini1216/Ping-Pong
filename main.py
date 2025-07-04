import pygame
import pickle
import numpy as np

# Game Constants
WIDTH, HEIGHT = 600, 300        # 🔄 Updated dimensions
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 50  # 🔽 Slightly shorter paddles
BALL_SIZE = 10
FPS = 60
MAX_SCORE = 3
NUM_BUCKETS = 12


# Load trained Q-table
with open("table_20000.pkl", "rb") as f:
    table = pickle.load(f)

# Safe discretization to avoid index errors
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

# Initialize Pygame
pygame.init()
pygame.event.pump()  # Ensure display initializes correctly
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ping Pong: AI vs You")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

# Game state
ball = [WIDTH // 2, HEIGHT // 2]
ball_vx, ball_vy = 2, 2  # Slower for smooth play
player_y = HEIGHT // 2
ai_y = HEIGHT // 2
player_score = 0
ai_score = 0
game_over = False

# Game loop
running = True
while running:
    screen.fill((0, 0, 0))

    # Draw paddles and ball
    pygame.draw.rect(screen, (255, 255, 255), (WIDTH - 20, player_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(screen, (255, 255, 255), (10, ai_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.ellipse(screen, (255, 255, 255), (*ball, BALL_SIZE, BALL_SIZE))

    # Display score
    score_text = font.render(f"AI: {ai_score}   You: {player_score}", True, (255, 255, 255))
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 10))
    pygame.display.flip()

    # End game if max score reached
    if game_over:
        pygame.time.wait(2000)
        winner = "AI" if ai_score == MAX_SCORE else "You"
        end_text = font.render(f"{winner} wins!", True, (255, 0, 0))
        screen.blit(end_text, (WIDTH // 2 - end_text.get_width() // 2, HEIGHT // 2))
        pygame.display.flip()
        pygame.time.wait(3000)
        break

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Human paddle movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        player_y -= 5
    if keys[pygame.K_DOWN]:
        player_y += 5
    player_y = max(0, min(player_y, HEIGHT - PADDLE_HEIGHT))

    # AI paddle movement
    if ball[0] < WIDTH // 2:
        # Ball far: center the paddle
        center_y = ai_y + PADDLE_HEIGHT // 2
        if center_y < HEIGHT // 2 - 5:
            ai_y += 2
        elif center_y > HEIGHT // 2 + 5:
            ai_y -= 2
    else:
        # Ball near: use Q-table
        state = get_state(ball, ball_vx, ball_vy, ai_y)
        q_values = table[state]
        action = np.argmax(q_values)
        if q_values[action] > 0.2:
            if action == 1:
                ai_y -= 4
            elif action == 2:
                ai_y += 4

    ai_y = max(0, min(ai_y, HEIGHT - PADDLE_HEIGHT))

    # Move the ball
    ball[0] += ball_vx
    ball[1] += ball_vy

    # Bounce off top/bottom
    if ball[1] <= 0 or ball[1] >= HEIGHT - BALL_SIZE:
        ball_vy *= -1

    # Player paddle collision
    if WIDTH - 20 <= ball[0] <= WIDTH - 10 and player_y < ball[1] < player_y + PADDLE_HEIGHT:
        ball_vx *= -1

    # AI paddle collision
    if 10 <= ball[0] <= 20 and ai_y < ball[1] < ai_y + PADDLE_HEIGHT:
        ball_vx *= -1

    # Scoring logic
    if ball[0] < 0:
        player_score += 1
        ball = [WIDTH // 2, HEIGHT // 2]
        ball_vx = 2
        ball_vy = 2
    elif ball[0] > WIDTH:
        ai_score += 1
        ball = [WIDTH // 2, HEIGHT // 2]
        ball_vx = -2
        ball_vy = 2

    if player_score == MAX_SCORE or ai_score == MAX_SCORE:
        game_over = True

    clock.tick(FPS)

pygame.quit()
