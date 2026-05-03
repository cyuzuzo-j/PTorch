import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

def get_mnist(batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    def collate_fn(batch):
        images, labels = zip(*batch)
        x = np.array(images, dtype=np.float32) / 255.0
        y = np.array(labels, dtype=np.int64)
        if x.ndim == 3: x = x[..., None]
        x = (x - MNIST_MEAN) / MNIST_STD
        return x, y

    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.view(-1))
    ])
    train = torchvision.datasets.MNIST('./dataset', train=True, download=True, collate_fn=collate_fn, transform=transform)
    test = torchvision.datasets.MNIST('./dataset', train=False, download=True, collate_fn=collate_fn,transform=transform)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=generator), \
           DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

def get_cifar10(batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        T.Lambda(lambda x: x.view(-1))
    ])
    train = torchvision.datasets.CIFAR10('./dataset', train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10('./dataset', train=False, download=True, transform=transform)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=generator), \
           DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


class MNISTDataModule:
    def __init__(self, batch_size: int = 64, seed: int = 42):
        self.batch_size = batch_size
        self.seed = seed
        self.train_loader, self.test_loader = get_mnist(batch_size, seed)

    def train_iterator(self):
        def _iterator():
            while True:
                for batch in self.train_loader:
                    yield batch
        return iter(_iterator())

    def val_dataloader(self):
        return self.test_loader

    def test_dataloader(self):
        return self.test_loader


class InfiniteCifarDataModule:
    def __init__(self, batch_size: int = 64, seed: int = 42):
        self.batch_size = batch_size
        self.seed = seed
        self.train_loader, self.test_loader = get_cifar10(batch_size, seed)

    def train_iterator(self):
        def _iterator():
            while True:
                for batch in self.train_loader:
                    yield batch
        return iter(_iterator())

    def val_dataloader(self):
        return self.test_loader

    def test_dataloader(self):
        return self.test_loader
