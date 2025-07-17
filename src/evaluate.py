import torch
import re
import pickle
import os
from src.model import GRUClassifier
from src.config import VOCAB_PATH, CHECKPOINT, MAX_LEN

def predict_text(text: str) -> float:
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Словарь не найден: {VOCAB_PATH}")
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    toks = re.findall(r"[\w']+|[.,!?;]", text.lower())
    indices = [vocab.get(tok, vocab.get('<UNK>')) for tok in toks][:MAX_LEN]
    indices += [vocab.get('<PAD>')] * (MAX_LEN - len(indices))
    x = torch.LongTensor(indices).unsqueeze(0)  # форма [1, MAX_LEN]

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Модель не найдена: {CHECKPOINT}")
    model = GRUClassifier(len(vocab))
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    return prob

def evaluate():
    import pickle
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from src.dataset import get_loader
    from src.config import TEST_CSV

    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    model = GRUClassifier(len(vocab))
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    test_loader = get_loader(TEST_CSV, VOCAB_PATH)
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            probs = torch.sigmoid(logits).numpy()
            y_pred.extend((probs > 0.5).astype(int))
            y_true.extend(y.numpy().astype(int))

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
