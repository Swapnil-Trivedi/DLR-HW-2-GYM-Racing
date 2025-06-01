from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(csv_path):
    data = pd.read_csv(csv_path, header=None).values
    X = data[:, :-3].astype(np.float32) / 255.0  # Normalize image pixels
    y = data[:, -3:].astype(np.float32)          # Steering, gas, brake
    X = X.reshape((-1, 96, 96, 3))                # Reshape to (N, 96, 96, 3)
    return X, y

def create_cnn_model():
    model = Sequential([
        Input(shape=(96, 96, 3)),
        Conv2D(32, kernel_size=5, strides=2, activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(64, kernel_size=3, strides=2, activation='relu'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='linear')  # Output: [steer, gas, brake]
    ])
    model.compile(optimizer=Adam(1e-4), loss='mse')
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