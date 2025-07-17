import torch
def save_checkpoint(model, optimizer, path):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, path)

def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state'])
