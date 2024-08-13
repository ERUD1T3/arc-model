import json
import logging
from typing import Dict, Tuple, Any

from shared.globals import TRAIN_CHLG_PATH, TRAIN_SOL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(chlg_path: str, sol_path: str) -> Dict[str, Tuple[Any, Any]]:
    """
    Load challenge and solution data from JSON files and return a dictionary
    mapping task IDs to their respective features (X) and labels (y).

    :param chlg_path: str: Path to the training challenges JSON file.
    :param sol_path: str: Path to the training solutions JSON file.
    :return: Dict[str, Tuple[Any, Any]]: Dictionary mapping task IDs to features and labels.

    """
    # Open the JSON files and load the data
    with open(chlg_path) as f_X, open(sol_path) as f_y:
        data_X = json.load(f_X)
        data_y = json.load(f_y)

    # Initialize an empty dictionary to store the task data
    task_data = {}

    # Iterate over each task in the training data
    for task_id, task_content in data_X.items():
        task_X = task_content  # Features for the task
        task_data[task_id] = (task_X, data_y[task_id])  # Pair features with their labels

    return task_data


def main() -> None:
    """
    Main function to load the data and print task information using logging.
    """
    # Load the data from the specified paths
    data = load_data(TRAIN_CHLG_PATH, TRAIN_SOL_PATH)

    # log the data
    logging.info(f"Loaded {len(data)} tasks from {TRAIN_CHLG_PATH} and {TRAIN_SOL_PATH}")
    logging.info(data)

    # Log each task's ID, features (X), and labels (y)
    for task_id, (X, y) in data.items():
        logging.info(f"Task ID: {task_id}")
        logging.info(f"Task X: {X}")
        logging.info(f"Task y: {y}")
        # Returning after the first task; remove return if you want to log all tasks
        return


# Execute the main function
if __name__ == "__main__":
    main()
