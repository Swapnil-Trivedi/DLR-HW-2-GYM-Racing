from cnn_model import load_data, create_cnn_model , plot_loss
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
# Load and split data
X, y = load_data("driving_data_more_data.csv")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Create and train CNN
model = create_cnn_model()
checkpoint = ModelCheckpoint("./cnn_model.keras", save_best_only=True)
history=model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=64,
          callbacks=[checkpoint])
plot_model(model, to_file="./cnn_model_architecture.png", show_shapes=True, show_layer_names=True)
print("CNN model saved to cnn_model.keras")
print("Model summary:")
model.summary()
#plot loss
plot_loss(history, title="CNN Training Loss", save_path="./cnn_training_loss_plot.png")