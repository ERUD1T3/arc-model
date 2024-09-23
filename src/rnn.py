import os
import csv
import tensorflow as tf


from src.shared.globals import *


DATASET_PATH = "../data/arc-agi_training_challenges_bpe.csv"

def load_dataset():
    padded_sequence = []

    with open(DATASET_PATH, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            # Read row as integers
            curr_seq = list(map(int, row[3].split(' ')))
            curr_seq_len = int(row[2])

            # Add padding at the beginning of the sequence
            padded_sequence.append(pad_sequences(curr_seq_len, curr_seq))


    # Flatten the list of padded sequences into a single sequence
    flattened_sequence = [token for seq in padded_sequence for token in seq]

    # Convert the flattened sequence to a TensorFlow dataset
    tf_dataset = tf.constant(flattened_sequence)
    ids_dataset = tf.data.Dataset.from_tensor_slices(tf_dataset)

    # Batch the dataset to create sequences of length SEQ_LEN
    sequences = ids_dataset.batch(SEQ_LEN, drop_remainder=True)

    # Map to input and target sequences
    dataset = sequences.map(split_input_target)

    return dataset


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def pad_sequences(curr_seq_len, curr_seq):
    # Calculate number of padding required for the current sequence
    num_pads = SEQ_LEN - curr_seq_len

    # Add padding at the beggining, followd by the sequence
    padded_sequence = [PAD] * num_pads + curr_seq
    return padded_sequence

def try_model(dataset, model):
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()  # the __init__ on 2.17 passes "self" by default.
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            rnn_units,
            return_sequences=True,
            return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
      x = inputs
      x = self.embedding(x, training=training)
      if states is None: # initialized the state with zeros so dont crash on "Try the model" step
          batch_size = tf.shape(x)[0]
          states = tf.zeros([batch_size, self.gru.units], dtype=tf.float32)
      x, states = self.gru(x, initial_state=states, training=training)
      x = self.dense(x, training=training)

      if return_state:
          return x, states
      else:
          return x

def show_dataset_examples(dataset):
    for input_example, target_example in dataset.take(3):
        print("Input :", input_example.numpy())
        print("Target:", target_example.numpy())
    print()



def main():
    dataset = load_dataset()
    # show_dataset_examples(dataset)
    
    # Batch size
    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    BUFFER_SIZE = 10000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    # Length of the vocab
    vocab_size = VOCAB_SIZE

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    model = MyModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)
    
    # try_model(dataset, model)
    model.summary()

    # Train model
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5") # added ".weights.h5" to fix issue

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS = 2

    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
    model.save('saved_model/rnn_model')

main()