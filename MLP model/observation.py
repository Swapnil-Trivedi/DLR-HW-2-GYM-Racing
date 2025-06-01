import numpy as np
import pandas as pd
from keras.models import load_model

# Load trained model
#model = load_model("./MLP model/mlp_model_more_data.keras", compile=False)
model = load_model("./CNN model/cnn_model.keras", compile=False)

# Load training data
data = pd.read_csv("driving_data_more_data.csv", header=None).values
X = data[:, :-3].astype(np.float32) / 255.0  # Normalize pixel values
y = data[:, -3:].astype(np.float32)          # Ground truth actions

# Reshape flattened images into (96, 96, 3)
X = X.reshape((-1, 96, 96, 3))

# Show predictions for the first 2 training samples
print("\nModel Predictions on Training Set Examples:\n")
for i in range(20):
    obs_input = np.expand_dims(X[i], axis=0)  # Add batch dimension
#    obs_input = np.expand_dims(X[i].flatten(), axis=0)  # Shape: (1, 27648)

    pred_action = model.predict(obs_input, verbose=0)[0]
    true_action = y[i]

    print(f"Sample {i + 1}:")
    print(f"  True Action      Steering: {true_action[0]:.2f}, Gas: {true_action[1]:.2f}, Brake: {true_action[2]:.2f}")
    print(f"  Predicted Action Steering: {pred_action[0]:.2f}, Gas: {pred_action[1]:.2f}, Brake: {pred_action[2]:.2f}\n")
