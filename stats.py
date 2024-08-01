import json
# import tensorflow as tf

train_challenges_path = 'data/arc-agi_training_challenges.json'
train_solutions_path = 'data/arc-agi_training_solutions.json'
eval_challenges_path = 'data/arc-agi_evaluation_challenges.json'
eval_solutions_path = 'data/arc-agi_evaluation_solutions.json'
test_challenges_path = 'data/arc-agi_test_challenges.json'

def number_of_tasks(path):
    f = open(path)
    data = json.load(f)
    return len(data)

def main():
    num_tasks_train = number_of_tasks(train_challenges_path)
    num_tasks_test = number_of_tasks(test_challenges_path)
    num_tasks_eval = number_of_tasks(eval_challenges_path)

    num_sol_train = number_of_tasks(train_solutions_path)
    num_sol_eval = number_of_tasks(eval_solutions_path)

    print(f'Num tasks in training:      {num_tasks_train} | Num solutions in training:      {num_sol_train}')
    print(f'Num tasks in evaluation:    {num_tasks_eval} | Num solutions in evaluation:    {num_sol_eval}')
    print(f'Num tasks in test:          {num_tasks_test}')

main()