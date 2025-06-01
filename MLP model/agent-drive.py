import numpy as np
import gymnasium as gym
from keras.models import load_model

# Load the trained Keras model
model = load_model("./MLP model/mlp_model_more_data.keras",compile=False)

# Create the CarRacing environment
env = gym.make("CarRacing-v3", render_mode="human", continuous=True)

try:
    obs, info = env.reset()
    done = False

    while True:
        # Preprocess observation
        obs_flat = obs.astype(np.float32).flatten() / 255.0
        obs_input = np.expand_dims(obs_flat, axis=0)  # shape (1, 27648)

        # Predict action using the model
        action = model.predict(obs_input, verbose=0)[0]  # shape (3,)
        action = np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])  # [steer, gas, brake]

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            obs, info = env.reset()

except KeyboardInterrupt:
    print("\nmanually stopped")

finally:
    env.close()
