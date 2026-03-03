"""
SST-2 Data Module — HuggingFace datasets backend.

Loads the GLUE/SST-2 dataset via the `datasets` library.

Returned batches are (x, y) where:
  x : np.int64 array of shape (batch, max_seq_len) – token indices
  y : np.int64 array of shape (batch,)              – binary labels
"""
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset


def _basic_tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


class SST2DataModule:
    """SST-2 via HuggingFace datasets (glue/sst2).

    Args:
        batch_size:   mini-batch size.
        max_seq_len:  sequences are truncated to this length.
        vocab_size:   vocabulary size (most-frequent tokens kept).
        seed:         random seed for DataLoader shuffling.
    """

    def __init__(
        self,
        batch_size: int = 64,
        max_seq_len: int = 64,
        vocab_size: int = 10_000,
        seed: int = 42,
    ):
        self.batch_size  = batch_size
        self.max_seq_len = max_seq_len
        self.vocab_size  = vocab_size
        self.seed        = seed

        raw = load_dataset("glue", "sst2")

        # Build vocab from training split
        counter: dict[str, int] = {}
        for ex in raw["train"]:
            for tok in _basic_tokenize(ex["sentence"]):
                counter[tok] = counter.get(tok, 0) + 1

        top = sorted(counter, key=counter.get, reverse=True)[: vocab_size - 2]
        self._stoi = {"<pad>": 0, "<unk>": 1}
        self._stoi.update({t: i + 2 for i, t in enumerate(top)})
        self._pad_idx = 0

        # Minimal vocab object that exposes __len__
        class _Vocab:
            def __init__(self, stoi):
                self._stoi = stoi
            def __len__(self):
                return len(self._stoi)
        self.vocab = _Vocab(self._stoi)

        def _encode(text: str):
            ids = [self._stoi.get(t, 1) for t in _basic_tokenize(text)]
            return torch.tensor(ids[: self.max_seq_len], dtype=torch.long)

        self.train_data = [(_encode(ex["sentence"]), ex["label"]) for ex in raw["train"]]
        self.val_data   = [(_encode(ex["sentence"]), ex["label"]) for ex in raw["validation"]]
        self.test_data  = self.val_data   # GLUE test labels are withheld

    # ── collate ────────────────────────────────────────────────────
    def _collate(self, batch):
        texts, labels = zip(*batch)
        padded = pad_sequence(texts, batch_first=True, padding_value=self._pad_idx)
        if padded.size(1) < self.max_seq_len:
            padded = torch.nn.functional.pad(
                padded, (0, self.max_seq_len - padded.size(1)), value=self._pad_idx
            )
        return padded.numpy(), np.array(labels, dtype=np.int64)

    # ── loaders ────────────────────────────────────────────────────
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            drop_last=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
            drop_last=False,
        )

    def train_iterator(self):
        loader = self.train_dataloader()
        def _it():
            while True:
                yield from loader
        return _it()
