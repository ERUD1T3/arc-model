import tensorflow as tf
import numpy as np

from src.modules import dataloader
from src.shared.globals import *

ds = dataloader.load_data(TRAIN_CHLG_PATH, TRAIN_SOL_PATH)

# Extract/Split Training data and Testing data from each task
for task in ds:
    task_id, task_X, task_y = task

    train_data = task_X['train']
    test_data = task_X['test']
    break


X_train = []
y_train = []

# Split training data into features and labels
for d in train_data:
    curr_x, curr_y = d
    X_train.append(curr_x)
    y_train.append(curr_y.flatten)

print(np.array(X_train).shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(3, 3)),
  tf.keras.layers.Dense(36, activation='relu'),
  tf.keras.layers.Dense(18, activation='relu'),
  tf.keras.layers.Dense(81)
])

# model.compile(optimizer='adam', loss='mse')

# model.fit(X_train, y_train, epochs=5)