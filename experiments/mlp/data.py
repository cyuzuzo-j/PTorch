import gzip
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, MNIST, VisionDataset
from tqdm import tqdm

import os
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


class DataModule(ABC):
    """Base class for dataset loading and preprocessing."""

    def __init__(
        self,
        batch_size: int = 64,
        normalize: bool = True,
        data_dir: str = "./dataset",
        overfit_batches: int = 0,
        preload: bool = False,
        seed: int = 42,
        grayscale: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.data_dir = data_dir
        self.preload = preload
        self.seed = seed
        self.grayscale = grayscale

        # Load train/val dataset
        full_dataset = self.dataset_cls(data_dir, train=True, download=True)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(self.seed)
        )
        if overfit_batches:
            self.train_dataset = Subset(self.train_dataset, range(overfit_batches * batch_size))

        # Load test dataset
        self.test_dataset = self.dataset_cls(data_dir, train=False, download=True)

        if self.preload:
            print("Preloading data...")

            def _preload(dataset, name):
                return [item for item in tqdm(dataset, desc=f"Preloading {name}")]

            self.train_dataset = _preload(self.train_dataset, "train")
            self.val_dataset = _preload(self.val_dataset, "validation")
            self.test_dataset = _preload(self.test_dataset, "test")
            print("Data preloaded.")

    @property
    @abstractmethod
    def dataset_cls(self) -> type[VisionDataset]:
        pass

    @property
    @abstractmethod
    def data_mean(self):
        pass

    @property
    @abstractmethod
    def data_std(self):
        pass

    def collate_fn(self, batch):
        images, labels = zip(*batch)
        x = np.array(images, dtype=np.float32) / 255.0
        y = np.array(labels, dtype=np.int64)

        if self.grayscale and x.ndim == 4 and x.shape[-1] == 3:
            # Convert RGB to grayscale by averaging channels
            x = np.mean(x, axis=-1, keepdims=True) # Resulting shape (batch, H, W, 1)

        if x.ndim == 3:
            x = x[..., None]  # add channel dim
        if self.normalize:
            x = (x - np.array(self.data_mean)) / np.array(self.data_std)
        return x, y

    def train_iterator(self):
        train_loader = self.train_dataloader()

        def _iterator():
            while True:
                for batch in train_loader:
                    yield batch

        return iter(_iterator())

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True,
        )


class MNISTDataModule(DataModule):
    """MNIST dataset module with standard preprocessing."""

    @property
    def dataset_cls(self):
        return MNIST

    @property
    def data_mean(self):
        return (0.1307,)

    @property
    def data_std(self):
        return (0.3081,)


class CIFAR10DataModule(DataModule):
    """CIFAR-10 dataset module with standard preprocessing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self, "grayscale", False):
            self._data_mean = (0.4734,) # Average of (0.4914, 0.4822, 0.4465)
            self._data_std = (0.2009,) # Average of (0.2023, 0.1994, 0.2010)
        else:
            self._data_mean = (0.4914, 0.4822, 0.4465)
            self._data_std = (0.2023, 0.1994, 0.2010)

    @property
    def dataset_cls(self):
        return CIFAR10

    @property
    def data_mean(self):
        return self._data_mean

    @property
    def data_std(self):
        return self._data_std


class CIFAR10_CModule(CIFAR10DataModule):
    """Corrupted CIFAR10 dataset (only affects the test set)"""

    class CorruptedCIFAR10(Dataset):
        def __init__(self, corruptions, labels):
            self.corruptions = corruptions
            self.labels = labels

        def __len__(self):
            return len(self.corruptions) * len(self.labels)

        def __getitem__(self, idx):
            corruption_idx = idx // len(self.labels)
            corruption_key = sorted(self.corruptions.keys())[corruption_idx]
            label_idx = idx % len(self.labels)
            return self.corruptions[corruption_key][label_idx], self.labels[label_idx]

    def __init__(
        self,
        batch_size: int = 64,
        normalize: bool = True,
        data_dir: str = "./dataset",
        preload: bool = False,
        seed: int = 42,
        grayscale: bool = False,
    ):
        # Initialize train/val using parent class, test will be overwritten below
        super().__init__(batch_size, normalize, data_dir=data_dir, preload=preload, seed=seed, grayscale=grayscale)

        # download the corrupted CIFAR10 test set if it doesn't exist
        import tarfile
        from pathlib import Path

        import requests

        url = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"
        path = Path(data_dir) / "CIFAR-10-C"
        path.mkdir(parents=True, exist_ok=True)
        file = path / "CIFAR-10-C.tar"
        if not file.exists():
            with requests.get(url, stream=True) as r:
                with open(file, "wb") as f:
                    f.write(r.content)

            with tarfile.open(file) as f:
                f.extractall(path.parent)

        # load files
        corruptions = {}
        labels = None
        for file in path.glob("*.npy"):
            name = file.stem
            if name == "labels":
                labels = np.load(file)
            else:
                corruptions[name] = np.load(file)

        self.test_dataset = self.CorruptedCIFAR10(corruptions, labels)


class HIGGSDataModule:
    """HIGGS particle physics dataset module."""

    class HiggsDataset(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    def __init__(
        self,
        batch_size: int = 64,
        normalize: bool = True,
        data_dir: str = "./dataset",
        preload: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.data_dir = data_dir
        self.seed = seed

        # download the Higgs dataset if it doesn't exist
        path = Path(data_dir) / "HIGGS"
        path.mkdir(parents=True, exist_ok=True)
        csv_file = path / "HIGGS.csv"
        if not csv_file.exists():
            print(f"{csv_file} does not exist. Obtaining it...")
            higgs_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
            gz_file = path / "HIGGS.csv.gz"
            if not gz_file.exists():
                print(f"Downloading {higgs_url}...")
                with requests.get(higgs_url, stream=True) as r:
                    r.raise_for_status()
                    with open(gz_file, "wb") as f:
                        shutil.copyfileobj(r.raw, f)
                print("Download complete.")
            else:
                print(f"{gz_file} already exists. Skipping download.")

            print(f"Extracting {gz_file} to {csv_file}...")
            with gzip.open(gz_file, "rb") as f_in:
                with open(csv_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print("Extraction complete.")

        # load the dataset
        data = np.loadtxt(csv_file, delimiter=",")
        x = data[:, 1:]
        y = data[:, 0].astype(np.int64)

        # compute mean and std
        self.data_mean = x.mean(axis=0)
        self.data_std = x.std(axis=0)

        # split the dataset
        # last 500k samples are used for testing
        x_test, y_test = x[-500_000:], y[-500_000:]
        x_train_val, y_train_val = x[:-500_000], y[:-500_000]

        split = int(0.9 * len(x_train_val))
        shuffle = np.random.default_rng(self.seed).permutation(len(x_train_val))

        x_train, y_train = x_train_val[shuffle[:split]], y_train_val[shuffle[:split]]
        x_val, y_val = x_train_val[shuffle[split:]], y_train_val[shuffle[split:]]

        self.train_dataset = x_train, y_train
        self.val_dataset = x_val, y_val
        self.test_dataset = x_test, y_test

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        if self.normalize:
            x = (x - self.data_mean) / self.data_std
        return x, y

    def train_iterator(self):
        train_loader = self.train_dataloader()

        def _iterator():
            while True:
                for batch in train_loader:
                    yield batch

        return iter(_iterator())

    def train_dataloader(self):
        return DataLoader(
            self.HiggsDataset(*self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.HiggsDataset(*self.val_dataset),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.HiggsDataset(*self.test_dataset),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True,
        )


class ShakespeareDataModule:
    """Shakespeare text dataset module for character-level language modeling."""

    class TextDataset(Dataset):
        def __init__(self, text, seq_len):
            self.text = text
            self.seq_len = seq_len

        def __len__(self):
            return len(self.text) - self.seq_len

        def __getitem__(self, idx):
            return self.text[idx : idx + self.seq_len], self.text[idx + 1 : idx + self.seq_len + 1]

    def __init__(
        self,
        batch_size: int = 64,
        seq_len: int = 64,
        data_dir: str = "./dataset",
        preload: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data_dir = data_dir
        self.seed = seed

        # download the Shakespeare dataset if it doesn't exist
        txt_file = Path(data_dir) / "shakespeare.txt"
        if not txt_file.exists():
            text_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with requests.get(text_url, stream=True) as r:
                with open(txt_file, "wb") as f:
                    f.write(r.content)

        # load the dataset
        with open(txt_file, "r") as f:
            text = f.read()

        # create vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        char_to_idx = {ch: i for i, ch in enumerate(chars)}

        # convert text to tensor
        text = np.array([char_to_idx[ch] for ch in text], dtype=np.int64)

        # last 20% samples are used for testing
        n = len(text)
        test_start = int(0.8 * n)
        train_end = int(0.9 * test_start)
        self.train_dataset = text[:train_end]
        self.val_dataset = text[train_end:test_start]
        self.test_dataset = text[test_start:]

    def collate_fn(self, batch):
        x, y = zip(*batch)
        return np.array(x, dtype=np.int64), np.array(y, dtype=np.int64)

    def train_iterator(self):
        train_loader = self.train_dataloader()

        def _iterator():
            while True:
                for batch in train_loader:
                    yield batch

        return iter(_iterator())

    def train_dataloader(self):
        return DataLoader(
            self.TextDataset(self.train_dataset, self.seq_len),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.TextDataset(self.val_dataset, self.seq_len),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.TextDataset(self.test_dataset, self.seq_len),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True,
        )


#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

def make_random_square_masks(inputs, size):
    is_even = int(size % 2 == 0)
    n,c,h,w = inputs.shape

    # seed top-left corners of squares to cutout boxes from, in one dimension each
    corner_y = torch.randint(0, h-size+1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w-size+1, size=(n,), device=inputs.device)

    # measure distance, using the center as a reference point
    corner_y_dists = torch.arange(h, device=inputs.device).view(1, 1, h, 1) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1, 1, 1, w) - corner_x.view(-1, 1, 1, 1)

    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)

    final_mask = mask_y * mask_x

    return final_mask

def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)

def set_random_state(seed, state):
    if seed is None:
        # If we don't get a data seed, then make sure to randomize the state using independent generator, since
        # it might have already been set by the model seed.
        import random
        torch.manual_seed(random.randint(0, 2**63))
    else:
        seed1 = 1000 * seed + state # just don't do more than 1000 epochs or else there will be overlap
        torch.manual_seed(seed1)

class InfiniteCifarLoader:
    """
    CIFAR-10 loader which constructs every input to be used during training during the call to __iter__.
    The purpose is to support cross-epoch batches (in case the batch size does not divide the number of train examples),
    and support stochastic iteration counts in order to preserve perfect linearity/independence.
    """

    def __init__(self, path, train=True, batch_size=500, aug=None, altflip=True, subset_mask=None, aug_seed=None, order_seed=None):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data = torch.load(data_path, map_location=device)
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate', 'cutout'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.altflip = altflip
        self.subset_mask = subset_mask if subset_mask is not None else torch.tensor([True]*len(self.images)).to(device)
        self.train = train
        self.aug_seed = aug_seed
        self.order_seed = order_seed

    def __iter__(self):

        # Preprocess
        images0 = self.normalize(self.images)
        # Pre-randomly flip images in order to do alternating flip later.
        if self.aug.get('flip', False) and self.altflip:
            set_random_state(self.aug_seed, 0)
            images0 = batch_flip_lr(images0)
        # Pre-pad images to save time when doing random translation
        pad = self.aug.get('translate', 0)
        if pad > 0:
            images0 = F.pad(images0, (pad,)*4, 'reflect')
        labels0 = self.labels

        # Iterate forever
        epoch = 0
        batch_size = self.batch_size

        # In the below while-loop, we will repeatedly build a batch and then yield it.
        num_examples = self.subset_mask.sum().item()
        current_pointer = num_examples
        batch_images = torch.empty(0, 3, 32, 32, dtype=images0.dtype, device=images0.device)
        batch_labels = torch.empty(0, dtype=labels0.dtype, device=labels0.device)
        batch_indices = torch.empty(0, dtype=labels0.dtype, device=labels0.device)

        while True:

            # Assume we need to generate more data to add to the batch.
            assert len(batch_images) < batch_size

            # If we have already exhausted the current epoch, then begin a new one.
            if current_pointer >= num_examples:
                # If we already reached the end of the last epoch then we need to generate
                # a new augmented epoch of data (using random crop and alternating flip).
                epoch += 1

                set_random_state(self.aug_seed, epoch)
                images1 = images0
                if pad > 0:
                    images1 = batch_crop(images0, 32)
                if self.aug.get('flip', False):
                    if self.altflip:
                        images1 = images1 if epoch % 2 == 0 else images1.flip(-1)
                    else:
                        images1 = batch_flip_lr(images1)
                if self.aug.get('cutout', 0) > 0:
                    images1 = batch_cutout(images1, self.aug['cutout'])

                set_random_state(self.order_seed, epoch)
                indices = (torch.randperm if self.train else torch.arange)(len(self.images), device=images0.device)

                # The effect of doing subsetting in this manner is as follows. If the permutation wants to show us
                # our four examples in order [3, 2, 0, 1], and the subset mask is [True, False, True, False],
                # then we will be shown the examples [2, 0]. It is the subset of the ordering.
                # The purpose is to minimize the interaction between the subset mask and the randomness.
                # So that the subset causes not only a subset of the total examples to be shown, but also a subset of
                # the actual sequence of examples which is shown during training.
                indices_subset = indices[self.subset_mask[indices]]
                current_pointer = 0

            # Now we are sure to have more data in this epoch remaining.
            # This epoch's remaining data is given by (images1[current_pointer:], labels0[current_pointer:])
            # We add more data to the batch, up to whatever is needed to make a full batch (but it might not be enough).
            remaining_size = batch_size - len(batch_images)

            # Given that we want `remaining_size` more training examples, we construct them here, using
            # the remaining available examples in the epoch.

            extra_indices = indices_subset[current_pointer:current_pointer+remaining_size]
            extra_images = images1[extra_indices]
            extra_labels = labels0[extra_indices]
            current_pointer += remaining_size
            batch_indices = torch.cat([batch_indices, extra_indices])
            batch_images = torch.cat([batch_images, extra_images])
            batch_labels = torch.cat([batch_labels, extra_labels])

            # If we have a full batch ready then yield it and reset.
            if len(batch_images) == batch_size:
                assert len(batch_images) == len(batch_labels)
                yield (batch_indices, batch_images, batch_labels)
                batch_images = torch.empty(0, 3, 32, 32, dtype=images0.dtype, device=images0.device)
                batch_labels = torch.empty(0, dtype=labels0.dtype, device=labels0.device)
                batch_indices = torch.empty(0, dtype=labels0.dtype, device=labels0.device)

class FiniteCifarLoader:
    def __init__(self, loader, num_samples):
        self.loader = loader
        self.num_samples = num_samples

    def __iter__(self):
        iterator = iter(self.loader)
        steps = self.num_samples // self.loader.batch_size
        for _ in range(steps):
            _, x, y = next(iterator)
            yield x.float().permute(0, 2, 3, 1).cpu().numpy(), y.cpu().numpy()

class InfiniteCifarDataModule:
    def __init__(self, batch_size=64, seed=42, data_dir="./dataset", **kwargs):
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir

    def train_iterator(self):
        # Standard CIFAR augmentation: horizontal flip, 4px reflect-pad random
        # crop, and 8px cutout. Without this the ViT memorizes the 50k training
        # images and overfits (large train/val gap). Val/test loaders build their
        # own InfiniteCifarLoader without `aug`, so they stay un-augmented.
        aug = {'flip': False, 'translate': 0, 'cutout': 0}
        loader = InfiniteCifarLoader(self.data_dir, train=True, batch_size=self.batch_size, aug=aug, aug_seed=self.seed, order_seed=self.seed)
        def _iter():
            for _, x, y in loader:
                yield x.float().permute(0, 2, 3, 1).cpu().numpy(), y.cpu().numpy()
        return _iter()

    def val_dataloader(self):
        loader = InfiniteCifarLoader(self.data_dir, train=False, batch_size=self.batch_size)
        return FiniteCifarLoader(loader, 10000)

    def test_dataloader(self):
        return self.val_dataloader()

