import torch, os, pickle
import torch.nn as nn
import matplotlib.pyplot as plt
from src.config import TRAIN_CSV, VAL_CSV, VOCAB_PATH, CHECKPOINT, LR, EPOCHS
from src.dataset import get_loader
from src.model import GRUClassifier
from src.utils import save_checkpoint

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
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y.squeeze())
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x.size(0)
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = criterion(model(x), y.squeeze())
                epoch_val_loss += loss.item() * x.size(0)
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch}/{EPOCHS}: train_loss={epoch_train_loss:.4f}, val_loss={epoch_val_loss:.4f}")
        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            save_checkpoint(model, optimizer, CHECKPOINT)
            print(f" Сохранена модель с val_loss={best_val:.4f}")

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train()
