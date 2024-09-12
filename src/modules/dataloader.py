import json
import os
from typing import Dict, Tuple, Generator, Any

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


# TODO: a function to convert any size grid to 30x30 grid which is the max size in dataset


def main() -> None:
    """
    Main function to load the data and log task information using a generator.
    """
    # Load and log the data task by task
    ds = load_data(TRAIN_CHLG_PATH, TRAIN_SOL_PATH)
    for task in ds:
        task_info(*task)
        # Process only one task for demonstration; remove this return statement to process all tasks


# Execute the main function
if __name__ == "__main__":
    main()
