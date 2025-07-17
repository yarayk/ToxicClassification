import torch, re, pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.config import MAX_LEN, BATCH_SIZE

class ToxicDataset(Dataset):
    def __init__(self, csv_path, vocab_path, max_len=MAX_LEN):
        self.df = pd.read_csv(csv_path)
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['comment'].iloc[idx]
        toks = re.findall(r"[\w']+|[.,!?;]", text.lower())
        indices = [self.vocab.get(tok, self.vocab['<UNK>']) for tok in toks][:self.max_len]
        indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        target = float(self.df['toxic'].iloc[idx])
        return torch.LongTensor(indices), torch.FloatTensor([target])

def get_loader(csv_path, vocab_path, batch_size=BATCH_SIZE, shuffle=False):
    ds = ToxicDataset(csv_path, vocab_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
