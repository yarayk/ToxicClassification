import torch.nn as nn
from src.config import EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, NUM_LAYERS,
                          batch_first=True, bidirectional=True, dropout=DROPOUT)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM * 2, 1)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        o, _ = self.gru(emb)
        out = o[:, -1, :]
        return self.fc(out).squeeze(1)

