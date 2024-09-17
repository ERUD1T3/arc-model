# globals.py
import logging

DATA_DIR = 'data/'
# set the paths to the datasets
TRAIN_CHLG_PATH = DATA_DIR + 'arc-agi_training_challenges.json'
TRAIN_SOL_PATH = DATA_DIR + 'arc-agi_training_solutions.json'
EVAL_CHLG_PATH = DATA_DIR + 'arc-agi_evaluation_challenges.json'
EVAL_SOL_PATH = DATA_DIR + 'arc-agi_evaluation_solutions.json'
TEST_CHLG_PATH = DATA_DIR + 'arc-agi_test_challenges.json'

# constants - token indices map to the colors in the COLORS list
COLORS = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'magenta', 'orange', 'cyan', 'maroon']
N_COLORS = len(COLORS)
# special tokens
END_ROW = N_COLORS  # end of row
STR_GRID = N_COLORS + 1  # start of grid
END_GRID = N_COLORS + 2  # end of grid
STR_SEQ = N_COLORS + 3  # start of sequence
END_SEQ = N_COLORS + 4  # end of sequence
INPUT_IND = N_COLORS + 5  # input indicator
OUTPUT_IND = N_COLORS + 6  # output indicator
STR_EXS = N_COLORS + 7  # start of examples
END_EXS = N_COLORS + 8  # end of examples
PAD = N_COLORS + 9  # padding
VOCAB = COLORS + [END_ROW, STR_GRID, END_GRID, STR_SEQ, END_SEQ, INPUT_IND, OUTPUT_IND, STR_EXS, END_EXS, PAD]
INIT_VOCAB_SIZE = len(VOCAB)
# data/bpe2048/arc-agi_training_challenges_bpe_vocab.json
VOCAB_SIZE = 2048
SEQ_LEN = 1500  # maximum sequence length with above vocab (


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
