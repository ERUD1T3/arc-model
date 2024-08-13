import logging

DATA_DIR = 'data/'
# set the paths to the datasets
TRAIN_CHLG_PATH = DATA_DIR + 'arc-agi_training_challenges.json'
TRAIN_SOL_PATH = DATA_DIR + 'arc-agi_training_solutions.json'
EVAL_CHLG_PATH = DATA_DIR + 'arc-agi_evaluation_challenges.json'
EVAL_SOL_PATH = DATA_DIR + 'arc-agi_evaluation_solutions.json'
TEST_CHLG_PATH = DATA_DIR + 'arc-agi_test_challenges.json'

# constants
COLORS = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'magenta', 'orange', 'cyan', 'maroon']

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
