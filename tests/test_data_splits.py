"""Train / val / test separation: val is a seeded 10% of the official train set, train the
other 90%, test the official test set. Samples are identified by image bytes. Uses
./dataset and skips when it is absent (no downloads in tests)."""
import itertools, os

import pytest, torch
from torch.utils.data import DataLoader, Subset

from experiments.shared.data import InfiniteCifarDataModule, InfiniteCifarLoader, MNISTDataModule

DATA_DIR = "./dataset"
HAVE = {"mnist": os.path.isdir(os.path.join(DATA_DIR, "MNIST")), "cifar": os.path.isfile(os.path.join(DATA_DIR, "test.pt"))}
_ids = lambda batches, n=None: {row.tobytes() for x, _ in itertools.islice(batches, n) for row in x}


@pytest.mark.parametrize("name", ["mnist", "cifar"])
def test_train_val_test_are_separate(name):  # reference ids are built straight from the source files
    if not HAVE[name]:
        pytest.skip(f"{name} not cached")
    if name == "mnist":
        ds = MNISTDataModule(batch_size=500, seed=43, data_dir=DATA_DIR)
        ref = lambda d: _ids(DataLoader(d, batch_size=1000, collate_fn=ds.collate_fn))
        ref_val, ref_test = ref(Subset(ds.val_dataset.dataset, ds.val_dataset.indices)), ref(type(ds.test_dataset)(DATA_DIR, train=False))
    else:
        ds = InfiniteCifarDataModule(batch_size=500, seed=43, data_dir=DATA_DIR)
        ref = lambda L, m: _ids([(L.normalize(L.images[m.to(L.images.device)]).float().permute(0, 2, 3, 1).cpu().numpy(), 0)])
        test = InfiniteCifarLoader(DATA_DIR, train=False)
        ref_val, ref_test = ref(InfiniteCifarLoader(DATA_DIR), ds.val_mask), ref(test, torch.ones(len(test.images), dtype=bool))
    val, test, train = _ids(ds.val_dataloader()), _ids(ds.test_dataloader()), _ids(ds.train_iterator(), 20)
    assert val == ref_val and len(val) > 4_000  # exactly the held-out indices
    assert test == ref_test  # all of the official test file
    assert len(train) > 9_000 and train.isdisjoint(val) and train.isdisjoint(test)


@pytest.mark.skipif(not HAVE["mnist"], reason="MNIST not downloaded")
def test_label_noise_only_touches_train():
    clean = MNISTDataModule(batch_size=64, seed=43, data_dir=DATA_DIR)
    noisy = MNISTDataModule(batch_size=64, seed=43, data_dir=DATA_DIR, label_noise=0.2)
    assert noisy.val_dataset.indices == clean.val_dataset.indices
    idx = range(0, len(clean.val_dataset), 97)
    assert [noisy.val_dataset[i][1] for i in idx] == [clean.val_dataset[i][1] for i in idx]
    flipped = sum(noisy.train_dataset[i][1] != clean.train_dataset[i][1] for i in range(2000))
    assert flipped > 0
