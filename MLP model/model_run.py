import numpy as np
import gymnasium as gym
from keras.models import load_model

# Load model
model = load_model("keras_bc_model.keras", compile=False)

# Create environment
env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
obs, info = env.reset()

while True:
    obs_flat = obs.flatten().astype(np.float32) / 255.0
    obs_input = np.expand_dims(obs_flat, axis=0)  # shape: (1, 27648)

    action = model.predict(obs_input, verbose=0)[0]  # shape: (3,)
    action = np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()
