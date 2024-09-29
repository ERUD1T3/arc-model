import json
from typing import Dict, Tuple, Union, List

import numpy as np

from src.shared.globals import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


class Tokenizer:
    def __init__(self, vocab_path: str):
        """
        Initializes the Tokenizer by loading the vocabulary from a JSON file.

        :param vocab_path: Path to the JSON file containing the vocabulary.
        """
        self.vocab_path = vocab_path
        self.id_to_token = self.load_vocab(vocab_path)
        self.token_to_id = {self._token_to_str(v): int(k) for k, v in self.id_to_token.items()}

    def load_vocab(self, path: str) -> Dict[int, Union[str, List[int]]]:
        """
        Loads the vocabulary from a JSON file.

        :param path: Path to the JSON file.
        :return: Dictionary mapping token IDs to tokens.
        """
        with open(path, 'r') as f:
            vocab = json.load(f)

        # Convert string keys back to integers
        vocab = {int(k): v for k, v in vocab.items()}
        return vocab

    def _token_to_str(self, token: Union[str, List[int]]) -> str:
        """
        Converts a token to its string representation for token_to_id mapping.

        :param token: The token, which could be a string or a list representing merged tokens.
        :return: String representation of the token.
        """
        if isinstance(token, list):
            return ' '.join(map(str, token))
        else:
            return str(token)

    def tokenize(self, sequence: List[Union[str, int]]) -> List[int]:
        """
        Tokenizes a sequence of tokens into token IDs.

        :param sequence: List of tokens (could be strings or integers).
        :return: List of token IDs.
        """
        token_ids = []
        idx = 0
        while idx < len(sequence):
            # Try to find the longest matching token in the vocabulary
            matched = False
            for length in range(len(sequence) - idx, 0, -1):
                sub_seq = sequence[idx:idx + length]
                sub_seq_str = self._token_to_str(sub_seq)
                if sub_seq_str in self.token_to_id:
                    token_ids.append(self.token_to_id[sub_seq_str])
                    idx += length
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Token '{sequence[idx]}' not found in vocabulary.")
        return token_ids

    def detokenize(self, token_ids: List[int]) -> List[Union[str, int]]:
        """
        Detokenizes a sequence of token IDs back into tokens.

        :param token_ids: List of token IDs.
        :return: List of tokens (could be strings or integers).
        """
        sequence = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id)
            if token is None:
                raise ValueError(f"Token ID '{token_id}' not found in vocabulary.")
            if isinstance(token, list):
                # If the token represents a merged pair, expand it recursively
                expanded_tokens = self._expand_merged_token(token)
                sequence.extend(expanded_tokens)
            else:
                sequence.append(token)
        return sequence

    def _expand_merged_token(self, token: List[int]) -> List[Union[str, int]]:
        """
        Recursively expands a merged token into its basic tokens.

        :param token: A token represented as a list (merged tokens).
        :return: A flat list of basic tokens.
        """
        sequence = []
        for sub_token_id in token:
            sub_token = self.id_to_token.get(sub_token_id)
            if isinstance(sub_token, list):
                sequence.extend(self._expand_merged_token(sub_token))
            else:
                sequence.append(sub_token)
        return sequence
