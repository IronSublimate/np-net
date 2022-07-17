import numpy as np
import torch

from torchvision import datasets, transforms


class ModelBase:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, lr: float):
        pass


class FullyConnected(ModelBase):
    def __init__(self, in_size: int, out_size: int):
        self.a = np.random.rand(in_size, out_size).astype(dtype=np.float32)
        self.grad_a = np.zeros((in_size, out_size), dtype=np.float32)
        self.b = np.random.rand(in_size).astype(dtype=np.float32)
        self.grad_b = np.zeros(in_size, dtype=np.float32)

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return self.m @ inp

    def backward(self, dy: np.ndarray):
        return 0


class ReLU(ModelBase):
    def __init__(self):
        self.mask = np.ndarray()

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        self.mask = (inp > 0).astype(np.float32)
        return self.mask * inp

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy / self.mask


def train(model, train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.numpy()
        target = target.item()
        print(data.shape, target)


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)

    train_kwargs = {'batch_size': 1}
    test_kwargs = {'batch_size': 1}

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    train(None, train_loader)


if __name__ == '__main__':
    main()
