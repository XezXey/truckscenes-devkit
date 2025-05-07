import torch
from torch.utils.data import Dataset

def get_random_dataset():
    random_dataset = RandomDataset()
    random_dataloader = torch.utils.data.DataLoader(
        random_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    return random_dataloader, random_dataset

class RandomDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        # returns a fresh Tensor each time
        return torch.rand(10)     