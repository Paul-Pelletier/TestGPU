import tensorflow as tf

# Define matrix size and batch size
matrix_size = 20000  # Large matrix
batch_size = 5000  # Divide workload into smaller batches

# Function for batched matrix multiplication
def batched_matmul(a, b, batch_size):
    result = []
    for i in range(0, matrix_size, batch_size):
        batch_a = a[i:i+batch_size]
        batch_result = tf.matmul(batch_a, b)
        result.append(batch_result)
    return tf.concat(result, axis=0)

# Generate random matrices
a = tf.random.normal([matrix_size, matrix_size])
b = tf.random.normal([matrix_size, matrix_size])

# Perform batched computation
result = batched_matmul(a, b, batch_size)
print("Batched computation complete.")
