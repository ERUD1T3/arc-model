import numpy as np
import tensorflow as tf

from src.modules.dataloader import load_data, task_info, plot_grid
from src.shared.globals import *

ds = load_data(TRAIN_CHLG_PATH, TRAIN_SOL_PATH)

# Extract/Split Training data and Testing data from each task
for task in ds:
    task_id, task_X, task_y = task
    train_data = task_X['train']
    test_data = task_X['test']
    # get info about the task
    task_info(*task)
    break

X_train = []
y_train = []

# Split training data into features and labels
for d in train_data:
    curr_x, curr_y = d
    X_train.append(curr_x.flatten())  # flatten to get same 1D shape matching input layer
    y_train.append(curr_y.flatten())  # flatten to get same 1D shape matching output layer

# Convert to numpy arrays
X_train = np.array(X_train)  # 5x3x3 to 5x9, batch_size x input_shape
y_train = np.array(y_train)
# Print the shapes of the training data
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(9,)),
    tf.keras.layers.Dense(36, activation='relu'),
    tf.keras.layers.Dense(18, activation='relu'),
    tf.keras.layers.Dense(81)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=2000)

# Predict the output for the test data
test_data = test_data.flatten()  # 3x3 to 9
test_data = np.array([test_data])  # simulate batch size of 1, 9 to 1x9
pred = model.predict(test_data)  # it expects batch_size x input_shape, 1x9
# Reshape the prediction to the original shape
pred = pred[0].reshape((9, 9))
# Plot the true grid
plot_grid(task_y, title='True Grid', show_plot=False, save_plot=True, folder_name='results')
# Plot the predicted grid
plot_grid(pred, title='Predicted Grid', show_plot=False, save_plot=True, folder_name='results')
# predict the output for the training data
pred_train = model.predict(X_train)
# plot the first training example
plot_grid(pred_train[0].reshape((9, 9)), title='Predicted Grid (Training Example 1)', show_plot=False, save_plot=True, folder_name='results')
plot_grid(y_train[0].reshape((9, 9)), title='True Grid (Training Example 1)', show_plot=False, save_plot=True, folder_name='results')