import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def load_data(csv_path):
    data = pd.read_csv(csv_path, header=None).values
    X = data[:, :-3] / 255.0  # Normalize pixel values
    y = data[:, -3:]          # Actions: [steering, gas, brake]
    return X.astype(np.float32), y.astype(np.float32)

def create_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(3, activation='linear')  # Continuous outputs
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model

def plot_loss(history, title="Training Loss", save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    
    # Plot validation loss if available
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
