import csv
import json
import os
from collections import Counter
from typing import Dict, Tuple, Generator, Any, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.shared.globals import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(chlg_path: str, sol_path: str) -> Generator[Tuple[str, Dict[str, np.ndarray], np.ndarray], None, None]:
    """
    Load challenge and solution data from JSON files and yield one task at a time.

    The function converts the 2D arrays in the JSON files to NumPy arrays and returns
    them as a tuple containing the task ID, features (X), and corresponding labels (y).

    :param chlg_path: str: Path to the training challenges JSON file.
    :param sol_path: str: Path to the training solutions JSON file.
    :yield: Tuple[str, Dict[str, np.ndarray], np.ndarray]: A tuple containing the task ID,
            a dictionary with "train" and "test" keys holding NumPy arrays for inputs,
            and a NumPy array of the corresponding labels.
    """
    # Open the JSON files and load the data
    with open(chlg_path) as f_X, open(sol_path) as f_y:
        data_X, data_y = json.load(f_X), json.load(f_y)

    # Iterate over each task in the training data
    for task_id, task_content in data_X.items():
        # Convert input and output arrays in "train" and "test" to NumPy arrays
        task_X = {
            "train": [(np.array(pair["input"]), np.array(pair["output"])) for pair in task_content["train"]],
            # putting the input and output in a tuple
            "test": np.array(task_content["test"][0]["input"])
            # only the input is needed for the test data
        }
        # Convert the solution data to a NumPy array
        task_y = np.array(data_y[task_id][0])  # Only one solution per task

        # Yield one task at a time
        yield task_id, task_X, task_y


def tokenize_grid(grid: np.ndarray) -> np.ndarray:
    """
    Tokenizes a grid efficiently without adding an END_ROW after the last row.

    :param grid: np.ndarray: A 2D NumPy array representing the grid.
    :return: np.ndarray: A 1D NumPy array of tokens representing the grid.
    """
    num_rows, num_cols = grid.shape

    # Ensure the grid is of integer type
    grid_int = grid.astype(int)

    # Flatten the grid to a 1D array
    grid_flat = grid_int.flatten()

    # Calculate positions where END_ROW tokens should be inserted (after each row except the last)
    positions = np.arange(num_cols, num_rows * num_cols, num_cols)

    # Insert END_ROW tokens at the calculated positions
    tokens_body = np.insert(grid_flat, positions, END_ROW)

    # Concatenate STR_GRID at the start and END_GRID at the end
    tokens = np.concatenate(([STR_GRID], tokens_body, [END_GRID]))

    return tokens


def detokenize_grid(tokens: np.ndarray) -> np.ndarray:
    """
    Converts tokens back to a 2D grid efficiently.

    :param tokens: np.ndarray: A 1D NumPy array of tokens representing the grid.
    :return: np.ndarray: A 2D NumPy array representing the grid.
    """
    # Remove the STR_GRID and END_GRID tokens
    tokens_body = tokens[1:-1]

    # Find positions of END_ROW tokens
    end_row_indices = np.where(tokens_body == END_ROW)[0]

    # Prepare split indices for np.split (positions after which to split)
    split_indices = end_row_indices + 1  # np.split splits at positions after the indices

    # Split tokens_body into rows
    rows = np.split(tokens_body, split_indices)

    # Remove END_ROW tokens from each row
    grid_rows = [row[row != END_ROW] for row in rows]

    # Convert list of rows into a 2D NumPy array
    grid = np.array(grid_rows, dtype=int)

    return grid


def tokenize_io_pair(X: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Tokenizes an input-output pair efficiently

    :param X: Tuple[np.ndarray, np.ndarray]: A tuple of 2D NumPy arrays representing the input and output grids.
    :return: np.ndarray: A 1D NumPy array of tokens representing the input-output pair.
    """
    # Tokenize the input and output grids
    tokens_in = tokenize_grid(X[0])
    tokens_out = tokenize_grid(X[1])

    # Concatenate STR_SEQ at the start and END_SEQ at the end
    tokens = np.concatenate(([INPUT_IND], tokens_in, [OUTPUT_IND], tokens_out))

    return tokens


def tokenize_examples(examples: list[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Tokenizes a list of input-output examples efficiently.

    :param examples: list[Tuple[np.ndarray, np.ndarray]]: A list of tuples of 2D NumPy arrays representing input-output pairs.
    :return: np.ndarray: A 1D NumPy array of tokens representing the examples.
    """
    # Tokenize each input-output pair
    tokens_list = [tokenize_io_pair(pair) for pair in examples]

    # Concatenate STR_EXS at the start and END_EXS at the end
    tokens = np.concatenate(([STR_EXS], *tokens_list, [END_EXS]))

    return tokens


def tokenize_task(task_X: Dict[str, any], task_y: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Tokenizes the features of a task efficiently.

    :param task_X: Dict[str, np.ndarray]: A dictionary containing "train" and "test" data.
        - "train" is a list of tuples of 2D NumPy arrays.
        - "test" is a 2D NumPy array of query input data.
    :param task_y: np.ndarray: A 2D NumPy array representing the solution/output data.
    :return: np.ndarray: A 1D NumPy array of tokens representing the task features.
            int: index where the solution starts in the tokens array.
    """
    # Tokenize the examples and test data
    tokens_examples = tokenize_examples(task_X["train"])
    tokens_query = tokenize_grid(task_X["test"])
    tokens_sol = tokenize_grid(task_y)

    # Concatenate the tokens
    tokens = np.concatenate(([STR_SEQ], tokens_examples, [INPUT_IND], tokens_query, [OUTPUT_IND]))

    # Calculate the index where the solution starts
    sol_start = len(tokens)

    # Concatenate the solution tokens
    tokens = np.concatenate((tokens, tokens_sol, [END_SEQ]))

    return tokens, sol_start


def task_training_seqs(task_X: Dict[str, any], task_y: np.ndarray) -> List[np.ndarray]:
    """
    Create autoregressive training sequences for a task, where each sequence adds one more token
    from the solution for training an autoregressive model.

    :param task_X: Dict[str, any]: A dictionary containing "train" and "test" data.
        - "train" is a list of tuples of 2D NumPy arrays (input and output grids).
        - "test" is a 2D NumPy array of query input data.
    :param task_y: np.ndarray: A 2D NumPy array representing the solution/output grid.
    :return: List[np.ndarray]: A list of NumPy arrays where each array is a training sequence.
    """
    # Tokenize the task features and get the index where the solution starts
    tokens, sol_start = tokenize_task(task_X, task_y)

    # Initialize the list of training sequences
    sequences = []

    # Get the total number of tokens
    total_tokens = len(tokens)

    # Generate sequences by adding one token from the solution at a time
    for i in range(sol_start + 1, total_tokens + 1):
        # Include the tokens up to the current token in the solution
        seq = tokens[:i]
        sequences.append(seq)

    return sequences


def add_suffix_to_filename(filename: str, suffix: str) -> str:
    """
    Adds a suffix before the file extension in a filename.

    :param filename: str: The original filename.
    :param suffix: str: The suffix to add.
    :return: str: The new filename with the suffix added.
    """
    base, ext = os.path.splitext(filename)
    return f"{base}{suffix}.csv"


def save_token_sequences(ds: Generator[Tuple[str, Dict[str, Any], np.ndarray], None, None], output_path: str) -> None:
    """
    Generates token sequences from the dataset and saves them to a CSV file.

    :param ds: Generator yielding task data.
    :param output_path: str: Path to the output CSV file.
    """

    total_sequences = 0

    # Open the CSV file for writing
    with open(output_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header
        csv_writer.writerow(['task_id', 'sequence_index', 'sequence_length', 'sequence'])

        for task in ds:
            task_id, task_X, task_y = task

            # Generate the token sequences
            sequences = task_training_seqs(task_X, task_y)

            total_sequences += len(sequences)

            # For each sequence, write a row to the CSV
            for idx, seq in enumerate(sequences):
                seq_length = len(seq)
                seq_str = ' '.join(map(str, seq.tolist()))
                csv_writer.writerow([task_id, idx, seq_length, seq_str])

            # Log the number of sequences
            logging.info(f"Task ID: {task_id}, Number of sequences: {len(sequences)}")

        logging.info(f"Total sequences: {total_sequences}")

    print(f"Token sequences saved to {output_path}")


# def merge_pair_in_sequences(sequences: List[List[int]], pair_to_merge: Tuple[int, int], new_token: int) -> List[
#     List[int]]:
#     """
#     Merges the specified pair in all sequences by replacing occurrences with the new token.
#
#     :param sequences: List of token sequences.
#     :param pair_to_merge: Tuple representing the token pair to merge.
#     :param new_token: The new token representing the merged pair.
#     :return: Updated list of token sequences.
#     """
#     new_sequences = []
#     for seq in sequences:
#         i = 0
#         new_seq = []
#         while i < len(seq):
#             if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair_to_merge:
#                 new_seq.append(new_token)
#                 i += 2  # Skip the next token since we've merged it
#             else:
#                 new_seq.append(seq[i])
#                 i += 1
#         new_sequences.append(new_seq)
#     return new_sequences

def merge_pair_in_sequences(sequences: List[List[int]], pair_to_merge: Tuple[int, int], new_token: int) -> List[
    List[int]]:
    """
    Merges the specified pair in all sequences by replacing occurrences with the new token efficiently.
    NOTE: only works with integer tokens

    :param sequences: List of token sequences.
    :param pair_to_merge: Tuple representing the token pair to merge.
    :param new_token: The new token representing the merged pair.
    :return: Updated list of token sequences.
    """
    # Convert the pair to merge into a string
    pair_str = ','.join(map(str, pair_to_merge))
    new_token_str = str(new_token)
    new_sequences = []

    for seq in sequences:
        # Convert the sequence into a comma-separated string
        seq_str = ','.join(map(str, seq))
        # Replace all occurrences of the pair with the new token string
        seq_str = seq_str.replace(pair_str, new_token_str)
        # Split the string back into tokens and convert to integers
        new_seq = list(map(int, seq_str.split(',')))
        new_sequences.append(new_seq)

    return new_sequences


def get_pair_frequencies(sequences: List[List[int]]) -> Dict[Tuple[int, int], int]:
    """
    Counts the frequency of each pair of consecutive tokens in the sequences efficiently.

    :param sequences: List of token sequences.
    :return: Dictionary with token pairs as keys and their frequencies as values.
    """
    pair_freqs = Counter()
    for seq in sequences:
        # Use zip to create pairs of consecutive tokens
        pairs = zip(seq, seq[1:])
        # Update the Counter with the pairs
        pair_freqs.update(pairs)
    return pair_freqs


def update_vocab(vocab: Dict[int, Any], pair_to_merge: Tuple[int, int], new_token: int) -> None:
    """
    Updates the vocabulary with the new token representing the merged pair.

    :param vocab: The current vocabulary.
    :param pair_to_merge: The pair being merged.
    :param new_token: The new token representing the merged pair.
    """
    # Ensure pair elements are native Python ints
    pair_to_merge = (int(pair_to_merge[0]), int(pair_to_merge[1]))
    vocab[int(new_token)] = pair_to_merge  # Map new token to the pair it represents


def convert_numpy_types(obj):
    """
    Recursively converts numpy types to native Python types in the given object.

    :param obj: The object to convert.
    :return: The object with numpy types converted to native Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(e) for e in obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(e) for e in obj]
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    else:
        return obj


def apply_bpe(sequences: List[np.ndarray], vocab_size: int, initial_vocab: Dict[int, Any]) -> Tuple[
    List[List[int]], Dict[int, Any]]:
    """
    Applies BPE to the sequences to increase vocabulary size up to vocab_size.

    :param sequences: List of token sequences.
    :param vocab_size: Desired vocabulary size.
    :param initial_vocab: The initial vocabulary mapping.
    :return: Tuple of updated sequences and the updated vocabulary.
    """
    vocab = initial_vocab.copy()
    current_vocab_size = max(vocab.keys()) + 1  # Next available token index

    while current_vocab_size < vocab_size:
        # Count the frequencies of token pairs
        pair_freqs = get_pair_frequencies(sequences)
        if not pair_freqs:
            break  # No more pairs to merge

        # Find the most frequent pair
        most_freq_pair = max(pair_freqs, key=pair_freqs.get)
        logging.info(f"Merging pair {most_freq_pair} with frequency {pair_freqs[most_freq_pair]}")

        # Assign a new token to this pair
        new_token = current_vocab_size
        current_vocab_size += 1

        # Merge the pair in all sequences
        sequences = merge_pair_in_sequences(sequences, most_freq_pair, new_token)

        # Update the vocabulary
        update_vocab(vocab, most_freq_pair, new_token)

    return sequences, vocab


def save_token_sequences_with_bpe(ds: Generator[Tuple[str, Dict[str, Any], np.ndarray], None, None], output_path: str,
                                  vocab_size: int) -> None:
    """
    Generates token sequences from the dataset, applies BPE, and saves them to a CSV file.

    :param ds: Generator yielding task data.
    :param output_path: str: Path to the output CSV file.
    :param vocab_size: int: Desired vocabulary size after BPE.
    """

    # Initial vocabulary: mapping from tokens to their string representations (for clarity)
    initial_vocab = {i: color for i, color in enumerate(COLORS)}
    special_tokens = {
        END_ROW: 'END_ROW',
        STR_GRID: 'STR_GRID',
        END_GRID: 'END_GRID',
        STR_SEQ: 'STR_SEQ',
        END_SEQ: 'END_SEQ',
        INPUT_IND: 'INPUT_IND',
        OUTPUT_IND: 'OUTPUT_IND',
        STR_EXS: 'STR_EXS',
        END_EXS: 'END_EXS',
        PAD: 'PAD'
    }
    initial_vocab.update(special_tokens)

    # Collect all sequences
    all_sequences = []
    task_sequence_indices = []  # Keep track of sequences per task
    task_ids = []

    for task in ds:
        task_id, task_X, task_y = task

        # Generate the token sequences
        sequences = task_training_seqs(task_X, task_y)

        # Keep track of indices
        start_idx = len(all_sequences)
        all_sequences.extend(sequences)
        end_idx = len(all_sequences)
        task_sequence_indices.append((task_id, start_idx, end_idx))
        task_ids.append(task_id)

        # Log the number of sequences
        logging.info(f"Task ID: {task_id}, Number of sequences: {len(sequences)}")

    # Log the total number of sequences
    logging.info(f"Total sequences: {len(all_sequences)}")

    # Apply BPE to all sequences
    sequences_after_bpe, final_vocab = apply_bpe(all_sequences, vocab_size, initial_vocab)

    # Save the sequences to CSV
    with open(output_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow(['task_id', 'sequence_index', 'sequence_length', 'sequence'])

        for task_id, start_idx, end_idx in task_sequence_indices:
            for idx in range(start_idx, end_idx):
                seq = sequences_after_bpe[idx]
                seq_length = len(seq)
                seq_str = ' '.join(map(str, seq))
                sequence_index = idx - start_idx  # Index within the task
                csv_writer.writerow([task_id, sequence_index, seq_length, seq_str])

    # Save the final vocabulary to a JSON file
    vocab_output_path = os.path.splitext(output_path)[0] + '_vocab.json'
    with open(vocab_output_path, 'w') as f:
        # Convert the entire vocabulary to native Python types
        json_vocab = {str(k): convert_numpy_types(v) for k, v in final_vocab.items()}
        json.dump(json_vocab, f)

    logging.info(f"Token sequences saved to {output_path}")
    logging.info(f"Final vocabulary saved to {vocab_output_path}")


def plot_grid(
        grid: np.ndarray,
        title: str = "Grid Plot",
        show_plot: bool = True,
        save_plot: bool = False,
        folder_name: str = "plots"
) -> None:
    """
    Plots a grid using the specified colors, with options to show or save the plot.

    :param grid: np.ndarray: A 2D NumPy array representing the grid.
    :param title: str: The title of the plot. Defaults to "Grid Plot".
    :param show_plot: bool: Whether to show the plot. Defaults to True.
    :param save_plot: bool: Whether to save the plot. Defaults to False.
    :param folder_name: str: The folder name where the plot will be saved. Defaults to "plots".
    """
    # Create a color map from the COLORS list
    cmap = mcolors.ListedColormap(COLORS)

    # Normalize the grid values to map to colors
    norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, len(COLORS) + 0.5), ncolors=len(COLORS))

    # Plot the grid
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.title(title)
    plt.colorbar(ticks=np.arange(len(COLORS)))

    # Get the number of rows and columns in the grid
    num_rows, num_cols = grid.shape

    # Set up the grid lines based on the actual shape of the grid
    plt.gca().set_xticks(np.arange(0.5, num_cols, 1), minor=True)
    plt.gca().set_yticks(np.arange(0.5, num_rows, 1), minor=True)

    # Enable gridlines and make sure they are on top
    plt.gca().grid(which="minor", color="white", linestyle='-', linewidth=1)
    plt.gca().tick_params(which="minor", size=0)
    plt.gca().set_axisbelow(False)

    if save_plot:
        # Ensure the folder exists
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Create a file name and save the plot
        file_path = os.path.join(folder_name, f"{title.replace(' ', '_')}.png")
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def task_info(*task_data: Tuple[str, Dict[str, Any], np.ndarray]) -> None:
    """
    Logs detailed information about a given task, including the number of training examples,
    their shapes, and the shape of the test data.

    :param task_data: Tuple containing task_id (str), task_X (dict), and task_y (np.ndarray).
                      - task_id (str): The identifier for the task.
                      - task_X (dict): A dictionary containing "train" and "test" data.
                          - "train" is a list of tuples of NumPy arrays.
                          - "test" is a NumPy array of test input data.
                      - task_y (np.ndarray): A NumPy array representing the solution/output data.
    """
    # Unpacking the tuple into individual variables
    task_id, task_X, task_y = task_data

    # Log task ID
    logging.info(f"Processing Task ID: {task_id}")

    # Extract and log training data information
    task_examples = task_X['train']
    logging.info(f"X (train): {len(task_examples)} examples")

    for i, (X, y) in enumerate(task_examples):
        logging.info(f"\tExample {i + 1}, X.shape: {X.shape}, y.shape: {y.shape}")

    # Extract and log test data information
    task_query = task_X['test']
    logging.info(f"X (test): {task_query.shape}")

    # Log the shape of the expected output
    logging.info(f"y: {task_y.shape}")


def main() -> None:
    """
    Main function to load the data and save token sequences to a file.
    """
    # # Load the data
    # ds = load_data(TRAIN_CHLG_PATH, TRAIN_SOL_PATH)
    # # for task in ds:
    # #     task_info(*task)
    # # Process only one task for demonstration; remove this return statement to process all tasks
    #
    # # Create the new file name with "_tok" added
    # chlg_tok_path = add_suffix_to_filename(TRAIN_CHLG_PATH, '_tok')
    # # Save the token sequences to the new file
    # save_token_sequences(ds, chlg_tok_path)

    # Load the data
    ds = load_data(TRAIN_CHLG_PATH, TRAIN_SOL_PATH)

    # Create the new file name with "_bpe" added
    chlg_tok_path = add_suffix_to_filename(TRAIN_CHLG_PATH, '_bpe')

    # Set the desired vocabulary size
    desired_vocab_size = 2000  # Adjust as needed

    # Save the token sequences with BPE applied
    save_token_sequences_with_bpe(ds, chlg_tok_path, desired_vocab_size)


# Execute the main function
if __name__ == "__main__":
    main()
