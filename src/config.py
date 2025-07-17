import os

# Пути
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
VAL_CSV = os.path.join(DATA_DIR, 'val.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.pkl')
CHECKPOINT = os.path.join(DATA_DIR, 'best_model.pt')

# Гиперпараметры
MAX_VOCAB = 20000
MAX_LEN = 100
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10

# fefe