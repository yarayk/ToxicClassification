import torch
import torch.nn as nn
from src.config import TRAIN_CSV, VAL_CSV, VOCAB_PATH, CHECKPOINT, LR, EPOCHS
from src.dataset import get_loader
from src.model import GRUClassifier
from src.utils import save_checkpoint
import pickle

def train():
    train_loader = get_loader(TRAIN_CSV, VOCAB_PATH, shuffle=True)
    val_loader = get_loader(VAL_CSV, VOCAB_PATH)

    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    model = GRUClassifier(len(vocab))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y.squeeze()).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{EPOCHS}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, CHECKPOINT)
            print(f"  → Сохранена модель с val_loss={best_val:.4f}")

if __name__ == '__main__':
    train()
