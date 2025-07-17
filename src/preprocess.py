import os, pandas as pd, pickle, re
from sklearn.model_selection import train_test_split
from collections import Counter
from src.config import DATA_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, VOCAB_PATH, MAX_VOCAB

def tokenize(text):
    return re.findall(r"[\w']+|[.,!?;]", text.lower())

def build_vocab(texts, max_vocab=MAX_VOCAB):
    cnt = Counter(tok for t in texts for tok in tokenize(t))
    most = [w for w, _ in cnt.most_common(max_vocab)]
    vocab = {w: i + 2 for i, w in enumerate(most)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def preprocess():
    path = os.path.join(DATA_DIR, 'labeled.csv')
    df = pd.read_csv(path)
    print("Колонки:", df.columns.tolist())

    train, temp = train_test_split(df, test_size=0.2, stratify=df['toxic'], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['toxic'], random_state=42)

    train.to_csv(TRAIN_CSV, index=False)
    val.to_csv(VAL_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    vocab = build_vocab(train['comment'].tolist())
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Словарь ({len(vocab)} токенов) сохранён в {VOCAB_PATH}")

if __name__ == '__main__':
    preprocess()
