import json
# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

train_challenges_path = 'data/arc-agi_training_challenges.json'
train_solutions_path = 'data/arc-agi_training_solutions.json'
eval_challenges_path = 'data/arc-agi_evaluation_challenges.json'
eval_solutions_path = 'data/arc-agi_evaluation_solutions.json'
test_challenges_path = 'data/arc-agi_test_challenges.json'

def number_of_tasks(path):
    f = open(path)
    data = json.load(f)
    return len(data)


def find_number_of_tasks():
    num_tasks_train = number_of_tasks(train_challenges_path)
    num_tasks_test = number_of_tasks(test_challenges_path)
    num_tasks_eval = number_of_tasks(eval_challenges_path)

    num_sol_train = number_of_tasks(train_solutions_path)
    num_sol_eval = number_of_tasks(eval_solutions_path)

    print(f'Num tasks in training:      {num_tasks_train} | Num solutions in training:      {num_sol_train}')
    print(f'Num tasks in evaluation:    {num_tasks_eval} | Num solutions in evaluation:    {num_sol_eval}')
    print(f'Num tasks in test:          {num_tasks_test}')
    print()


def get_grid_size(grid):
    return len(grid) * len(grid[0])


def find_max_min_grid(path):
    f = open(path)
    data = json.load(f)

    grid_sizes = []

    for task in data.values():
        grid_sizes.extend([get_grid_size(sample['input']) for sample in task['train']])
        grid_sizes.extend([get_grid_size(sample['output']) for sample in task['train']])
        if 'test' in task:
            grid_sizes.extend([get_grid_size(sample['input']) for sample in task['test']])

    min_grid_size = min(grid_sizes)
    max_grid_size = max(grid_sizes)

    print(f'Minimum grid size: {min_grid_size}')
    print(f'Maximum grid size: {max_grid_size}')
    print()


def find_max_min_values(path):
    f = open(path)
    data = json.load(f)

    min_val = float('inf')
    max_val = -1

    for task in data.values():
        for sample in task['train']:
            curr_min_in = min(min(sample['input']))
            curr_max_in = max(max(sample['input']))

            curr_min_out = min(min(sample['output']))
            curr_max_out = max(max(sample['output']))

            temp_min = min(curr_min_in, curr_min_out)
            temp_max = max(curr_max_in,curr_max_out)

            min_val = min(min_val, temp_min)
            max_val = max(max_val, temp_max)
            
        if 'test' in task:
            for sample in task['test']:
                curr_min_test = min(min(sample['input']))
                curr_max_test = max(max(sample['input']))
                
                temp_min = min(temp_min, curr_min_test)
                temp_max = max(temp_max, curr_max_test)
                min_val = min(min_val, temp_min)
                max_val = max(max_val, temp_max)



    print(f'Min value in dataset: {min_val}')
    print(f'Max value in dataset: {max_val}')
    print()


def draw_matrix(matrix):
    colors = {
        0: 'black',
        1: 'blue',
        2: 'red',
        3: 'green',
        4: 'yellow',
        5: 'purple',
        6: 'pink',
        7: 'orange'
    }

    fig, ax = plt.subplots()

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            color = colors.get(matrix[i][j], 'white')
            rect = plt.Rectangle((j, len(matrix)-1-i), 1, 1,
                                facecolor=color, edgecolor='gray')
            ax.add_patch(rect)

    ax.set_xlim(0, len(matrix))
    ax.set_ylim(0, len(matrix))
    ax.set_aspect('equal')
    ax.axis('off')

    plt.show()

def get_avg_number_examples(path):
    f = open(path)
    data = json.load(f)

    examples = []
    for task in data.values():
        examples.append(len(task['train']))

    return sum(examples)/len(examples)
    
def find_avg_number_examples():
    avg_examp_train = get_avg_number_examples(train_challenges_path)
    avg_examp_eval = get_avg_number_examples(eval_challenges_path)
    avg_examp_test = get_avg_number_examples (test_challenges_path)

    print(f'Avg num example in training:      {avg_examp_train}')
    print(f'Avg num examples in evaluation:    {avg_examp_eval}')
    print(f'Avg num examples in test:          {avg_examp_test}')
    print()

def draw_test():
    # example "025d127b" test set
    return [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
            [0, 4, 0, 0, 0, 0, 0, 4, 0, 0],
            [0, 0, 4, 0, 0, 0, 0, 0, 4, 0],
            [0, 0, 0, 4, 0, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 4, 4, 4, 4, 4, 4],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

def main():
    find_number_of_tasks()
    find_max_min_grid(eval_challenges_path)
    find_max_min_values(eval_challenges_path)
    find_avg_number_examples()
    draw_matrix(draw_test())
main()