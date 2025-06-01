from model import load_data, create_model , plot_loss
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from sklearn.model_selection import train_test_split


# Load dataset
X, y = load_data("driving_data_more_data.csv")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = create_model(input_dim=X.shape[1])

# Train
checkpoint = ModelCheckpoint("mlp_model_more_data.keras", save_best_only=True)
history=model.fit(X_train, y_train, validation_data=(X_val, y_val),
          batch_size=70, epochs=50, callbacks=[checkpoint])
plot_model(model, to_file="mlp_model_architecture.png", show_shapes=True, show_layer_names=True)
print("Keras model saved to keras_mlp_model_more_data.keras")
print("model summary:")
model.summary()
# Plot training loss
plot_loss(history, title="Training Loss", save_path="mlp_training_loss_plot.png")

