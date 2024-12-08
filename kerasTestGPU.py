import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

# Generate synthetic dataset
x_train = tf.random.normal([1000, 1])  # Simplified to 1 feature for easy visualization
y_train = 3 * x_train**2 + tf.random.normal([1000, 1])  # y = 3x + noise

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(1,)),  # Input shape for 1 feature
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model and measure training time
start_time = time.time()
history = model.fit(x_train, y_train, epochs=20, batch_size=100, verbose=0)  # Collect training loss
end_time = time.time()

print("Training Time on GPU: {:.4f} seconds".format(end_time - start_time))

# Make predictions
y_pred = model.predict(x_train)

# Extract loss values from the history object
loss_values = history.history['loss']
epochs = range(1, len(loss_values) + 1)

# Plotting
plt.figure(figsize=(12, 10))

# Plot 1: Training Data and Model Predictions
plt.subplot(2, 1, 1)
plt.scatter(x_train.numpy(), y_train.numpy(), label="Training Data", alpha=0.6)
plt.scatter(x_train.numpy(), y_pred, label="Model Predictions", color="red", alpha=0.6)
plt.title("Training Data and Model Predictions")
plt.xlabel("Input Feature (x_train)")
plt.ylabel("Target Value (y_train / y_pred)")
plt.legend()
plt.grid()

# Plot 2: Loss Function Across Epochs
plt.subplot(2, 1, 2)
plt.plot(epochs, loss_values, label="Training Loss", marker='o')
plt.title("Loss Function Across Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

# Show the plots
plt.tight_layout()
plt.show()
