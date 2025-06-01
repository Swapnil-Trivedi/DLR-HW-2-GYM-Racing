import gymnasium as gym
import pygame
import numpy as np
import time
import csv


env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
obs, info = env.reset()

pygame.init()
print("Use arrow keys to drive. Press ESC to quit.")

running = True
clock = pygame.time.Clock()

action = np.array([0.0, 0.0, 0.0])  # [steer, gas, brake]

data = []  # list to collect (observation, action)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break

    keys = pygame.key.get_pressed()
    action = np.array([0.0, 0.0, 0.0])

    if keys[pygame.K_LEFT]:
        action[0] = -1.0
    if keys[pygame.K_RIGHT]:
        action[0] = 1.0
    if keys[pygame.K_UP]:
        action[1] = 1.0
    if keys[pygame.K_DOWN]:
        action[2] = 0.8

    # Save flattened observation and action
    obs_flat = obs.flatten()  # shape: (27648,)
    row = np.concatenate([obs_flat, action])  # shape: (27648 + 3,)
    data.append(row)

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

    clock.tick(60)  # Limit to 60 FPS

# Save data to CSV
filename = "driving_data_more_data_1.csv"
print(f"\nSaving data to {filename} ...")
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)

print("Done! Data saved.")
env.close()
pygame.quit()
