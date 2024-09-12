import numpy as np

# Importing necessary functions from other modules for data loading and visualization
from src.modules.dataloader import load_data
from src.shared.globals import *


def generate_n_gram(X: list[np.ndarray], y: np.ndarray) -> list[np.ndarray]:
    """
    Generate n-grams by appending the current sequence of labels (y) to the existing
    list of input arrays (X). This helps in creating a sequence model input format.
    
    :param X: List of NumPy arrays representing the input data sequences.
    :param y: NumPy array of the target sequence (labels).
    :return: List of NumPy arrays where each entry is the concatenated sequence.
    """
    n_gram = X.copy()  # Initialize n-grams with a copy of the input sequences

    for i in range(0, len(y)):
        curr_cell = y[i]  # current cell
        curr_sequence = np.append(n_gram[-1], curr_cell)

        n_gram.append(curr_sequence)
    return n_gram


def convert_to_sequence(task_X: list) -> list[np.ndarray]:
    """
    Convert each example in the task data from 2D arrays (input, output) into a
    flattened sequence by concatenating input and output. This prepares the data
    for sequence-based models.
    
    :param task_X: List of tuples containing 2D input and output arrays.
    :return: List of 1D NumPy arrays where each array is a concatenated sequence of input and output.
    """
    X = []
    for example in task_X:
        x, y = example  # Unpack the input and output arrays

        x = x.reshape(-1)  # Flatten the input array
        y = y.reshape(-1)  # Flatten the output array

        sequence = np.concatenate((x, y))  # Concatenate input and output into a single sequence
        X.append(sequence)  # Add the sequence to the list

    return X


def test_n_gram(n_gram: list[np.ndarray]) -> None:
    """
    Not sure ways to test, but here is something.

    Observations from this "test": 
        - this test is ran in only one task, which is 007bbfb7, https://arcprize.org/play?task=007bbfb7
        - we know this task has:
            - 5 examples
                - same input sizes, (3x3)
                - same output sizes, (9x9)
            - 1 test, (3x3)
            - 1 solution, (9x9)

        - the length of n-gram should be (e + t + s), where:
                    "e" is the number of examples, "t" is the number of tests, and  "s" is the number of cells in solution.
                    so, len(n_gram) = (e + t + s) = 5 + 1 + (9x9) = 87

        - when printing the length of example[from 1 to 5], it should give us a length of (3x3) + (9x9) = 90.
        - when printing the length of example[from 6 to len(n_gram)], it should give us:
            - example[6] --> len(test). Because "test" was added at the end of X, after all examples.
            - example[7] --> len(example[6]) + 1. Because we start adding "y" (solution) cells one by one
            - example[8] --> len(example[7]) + 1.
                ...
    """

    print("\nExample # --> length/size \n---------------------------")
    for i in range(len(n_gram)):
        print(f"Example {i + 1} --> ", len(n_gram[i]))


def main() -> None:
    """
    Main function to load challenge and solution data, process each task,
    and demonstrate data transformations like sequence conversion and n-gram generation.
    """
    # Load challenge and solution data using predefined paths from globals
    ds = load_data(chlg_path=TRAIN_CHLG_PATH, sol_path=TRAIN_SOL_PATH)

    # Iterate over each task
    for task in ds:
        # Extract task's: ID, X (train, test) and y (test solution) data
        task_id, task_X, task_y = task

        # Split X into train data and test data
        train_data = task_X['train']
        test_data = task_X['test']

        # Convert train_data from a list of tuples (2D, 2D) to a list of 1D NumPy arrays
        X = convert_to_sequence(train_data)
        # Reshape test_data from 2D to 1D, and add it at the end of X
        X.append(test_data.reshape(-1))

        # Reshape the solution from 2D to 1D to match the input format
        task_y = task_y.reshape(-1)

        # Generate n-grams from the training data and labels
        n_grams = generate_n_gram(X, task_y)

        test_n_gram(n_grams)
        # task_info(task_id, task_X, task_y)

        return  # Remove this line to process all tasks


# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()
