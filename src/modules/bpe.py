from collections import Counter
from typing import Dict, Tuple, Any, List

import numpy as np

from src.shared.globals import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def merge_pair_in_sequences(
        sequences: List[List[int]], pair_to_merge: Tuple[int, int], new_token: int
) -> List[List[int]]:
    """
    Merges the specified pair in all sequences by replacing occurrences with the new token.

    :param sequences: List of token sequences.
    :param pair_to_merge: Tuple representing the token pair to merge.
    :param new_token: The new token representing the merged pair.
    :return: Updated list of token sequences.
    """
    new_sequences = []
    for seq in sequences:
        i = 0
        new_seq = []
        while i < len(seq):
            if (
                    i < len(seq) - 1
                    and seq[i] == pair_to_merge[0]
                    and seq[i + 1] == pair_to_merge[1]
            ):
                # Merge the pair
                new_seq.append(new_token)
                i += 2  # Skip the next token
            else:
                new_seq.append(seq[i])
                i += 1
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
    # log the initial vocab
    logging.info(f"Initial vocabulary: {vocab}")
    # log the initial vocab size and max token
    logging.info(f"Initial vocab size: {len(vocab)}")
    max_token = max(vocab.keys())
    logging.info(f"Max token: {max_token}")
    current_vocab_size = int(max_token) + 1  # Next available token index

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
        # log the new token
        logging.info(f"New produced token: {new_token}")

        # Merge the pair in all sequences
        sequences = merge_pair_in_sequences(sequences, most_freq_pair, new_token)

        # Update the vocabulary
        update_vocab(vocab, most_freq_pair, new_token)

    return sequences, vocab
