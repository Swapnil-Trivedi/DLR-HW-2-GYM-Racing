import numpy as np
import gymnasium as gym
from keras.models import load_model

# Load the CNN model
model = load_model("./CNN model/cnn_model.keras", compile=False)

# Create the environment
env = gym.make("CarRacing-v3", render_mode="human", continuous=True)

try:
    obs, info = env.reset()

    while True:
        # Preprocess observation for CNN input
        obs_input = np.expand_dims(obs.astype(np.float32) / 255.0, axis=0)

        # Predict action
        action = model.predict(obs_input, verbose=0)[0]
        action = np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            obs, info = env.reset()

except KeyboardInterrupt:
    print("\nstopped manually.")

finally:
    env.close()
