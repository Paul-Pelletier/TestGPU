import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the SVI function
def svi(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + tf.sqrt((k - m)**2 + sigma**2))

# Generate synthetic data for calibration
np.random.seed(42)
k_data = np.linspace(-1, 1, 100)  # Log-moneyness
true_params = [0.04, 0.2, -0.4, 0.0, 0.3]  # a, b, rho, m, sigma
w_data = (
    true_params[0]
    + true_params[1]
    * (true_params[2] * (k_data - true_params[3]) + np.sqrt((k_data - true_params[3])**2 + true_params[4]**2))
)
w_data += np.random.normal(0, 0.01, size=w_data.shape)  # Add noise

# Convert to TensorFlow tensors
k = tf.constant(k_data, dtype=tf.float32)
w = tf.constant(w_data, dtype=tf.float32)

# Initialize SVI parameters (randomized)
params = tf.Variable([0.05, 0.1, -0.1, 0.0, 0.2], dtype=tf.float32)  # a, b, rho, m, sigma

# Define the loss function (Mean Squared Error)
def loss_fn():
    w_pred = svi(k, *params)
    return tf.reduce_mean((w_pred - w)**2)

# Use Adam optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.1)

# Train the model
loss_history = []
for step in range(100):  # Number of iterations
    with tf.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, [params])
    optimizer.apply_gradients(zip(grads, [params]))
    loss_history.append(loss.numpy())
    
    # Log progress
    if step % 1 == 0:
        print(f"Step {step}, Loss: {loss.numpy():.6f}, Params: {params.numpy()}")

# Final parameters
print("Calibrated parameters:", params.numpy())

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(k_data, w_data, label="Synthetic Data (Noisy)", color="blue")
plt.plot(k_data, svi(k, *params).numpy(), label="Fitted SVI Curve", color="red", linewidth=2)
plt.xlabel("Log-moneyness (k)")
plt.ylabel("Total Variance (w)")
plt.title("SVI Calibration")
plt.legend()
plt.grid()
plt.show()
