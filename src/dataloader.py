import json

train_challenges_path = 'data/arc-agi_training_challenges.json'
train_solutions_path = 'data/arc-agi_training_solutions.json'
eval_challenges_path = 'data/arc-agi_evaluation_challenges.json'
eval_solutions_path = 'data/arc-agi_evaluation_solutions.json'
test_challenges_path = 'data/arc-agi_test_challenges.json'


def load_data(training_path, solution_path):
    f_X = open(training_path)
    f_y = open(solution_path)

    data_X = json.load(f_X)
    data_y = json.load(f_y)
    
    task_data = {}

    for task_id, task_content in data_X.items():

        task_X = task_content
        task_data[task_id] = (task_X, data_y[task_id])
    
    return task_data

def main():

    data = load_data(train_challenges_path, train_solutions_path)

    # for task_id, (X, y) in data.items():
    #     print(f"Task ID: {task_id}")
    #     print(f"Task X: {X}")
    #     print(f"Task y: {y}")
    #     return

main()