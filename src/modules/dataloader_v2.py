import numpy as np

from src.modules.dataloader import load_data, task_info, plot_grid
from src.shared.globals import *

def generate_n_gram(X: list[np.ndarray], y: np.ndarray):
    n_gram = []

    for i in range(0, len(y)):
        new_number = y[:i + 1]
        print(f"\n\n\n i: {i}", new_number)


    return

def convert_to_sequence(task_X: list) -> list[np.ndarray]:
    X = []
    for example in task_X:
        x, y = example
        
        x = x.reshape(-1)
        y = y.reshape(-1)

        sequence = np.concatenate((x, y))
        X.append(sequence)

    return X


def main() -> None:
    ds = load_data(chlg_path=TRAIN_CHLG_PATH, sol_path=TRAIN_SOL_PATH)

    for task in ds:
        task_id, task_X, task_y = task

        train_data = task_X['train']
        test_data = task_X['test'].reshape(-1)
        
        

        X = convert_to_sequence(train_data)
        X.append (test_data)

        y = task_y.reshape(-1)

        
        generate_n_gram(X=X, y=y)
        
        # task_info(*task)
        break

    X_train = []
    y_train = []

# Execute the main function
if __name__ == "__main__":
    main()

