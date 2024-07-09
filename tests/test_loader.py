import torch
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.data = torch.rand(n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx]


def test_loader():
    dataset = RandomDataset(100)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    for x in loader:
        y = x ** 2
        assert isinstance(y, torch.Tensor)
