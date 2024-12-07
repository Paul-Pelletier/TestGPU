import tensorflow as tf
import time

# Enable dynamic memory growth
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Adjust matrix size for GPU memory
matrix_size = 10000  # Fits within memory constraints

# Generate random matrices
a = tf.random.normal([matrix_size, matrix_size], dtype=tf.float16)
b = tf.random.normal([matrix_size, matrix_size], dtype=tf.float16)

# Perform matrix multiplication and measure time
start_time = time.time()
c = tf.matmul(a, b)
end_time = time.time()

import numpy as np
start_time1 = time.time()
c1 = np.random.normal(0,1,(matrix_size,matrix_size))
d1 = np.random.normal(0,1,(matrix_size,matrix_size))
e1 = np.matmul(c1,d1)
end_time1 = time.time()
print("Computation complete! Time taken GPU: {:.4f} seconds".format(end_time - start_time))
print("Computation complete! Time taken CPU: {:.4f} seconds".format(end_time1 - start_time1))
