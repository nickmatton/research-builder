"""
Data loader for the Transformer model (Attention Is All You Need).

Implements batching by approximate sequence length, targeting ~25,000
source tokens and ~25,000 target tokens per batch, as described in
Section 5.1 of the paper.

Usage:
    from dataloader import TranslationDataset, create_dataloader

    dataset = TranslationDataset("en_de_tokenized.pt")
    loader = create_dataloader(dataset, tokens_per_batch=25000)

    for batch in loader:
        src = batch['src']        # (batch_size, max_src_len) LongTensor, padded
        tgt = batch['tgt']        # (batch_size, max_tgt_len) LongTensor, padded
        src_mask = batch['src_mask']  # (batch_size, 1, max_src_len) BoolTensor
        tgt_mask = batch['tgt_mask']  # (batch_size, 1, max_tgt_len) BoolTensor
        ntokens = batch['ntokens']    # number of non-pad target tokens
"""

import os
import random
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

# Special token IDs (must match tokenizer training)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class TranslationDataset(Dataset):
    """Dataset of tokenized parallel sentence pairs."""

    def __init__(self, data_path):
        """
        Args:
            data_path: Path to .pt file containing tokenized data dict with
                       'src', 'tgt', 'metadata' keys.
        """
        data = torch.load(data_path, weights_only=False)
        self.src = data["src"]  # list of list of int
        self.tgt = data["tgt"]  # list of list of int
        self.metadata = data["metadata"]

        assert len(self.src) == len(self.tgt), "Source and target must have same length"

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {
            "src": self.src[idx],
            "tgt": self.tgt[idx],
        }

    @property
    def num_pairs(self):
        return len(self.src)

    @property
    def vocab_size(self):
        return self.metadata.get("vocab_size", None)


class TokenBatchSampler(Sampler):
    """
    Sampler that creates batches by approximate sequence length,
    targeting a specific number of tokens per batch.

    This implements the batching strategy from Section 5.1:
    "Sentence pairs were batched together by approximate sequence length.
     Each training batch contained a set of sentence pairs containing
     approximately 25000 source tokens and 25000 target tokens."

    Strategy:
    1. Sort all sentence pairs by source length (primary) and target length (secondary)
    2. Create pools of ~100 consecutive sorted examples
    3. Within each pool, create batches that fit within the token budget
    4. Shuffle the order of batches (not within batches)
    """

    def __init__(self, dataset, tokens_per_batch=25000, shuffle=True, pool_factor=100):
        """
        Args:
            dataset: TranslationDataset instance
            tokens_per_batch: Target number of source (and target) tokens per batch
            shuffle: Whether to shuffle batch order each epoch
            pool_factor: How many batches worth of examples to sort at once
        """
        self.dataset = dataset
        self.tokens_per_batch = tokens_per_batch
        self.shuffle = shuffle
        self.pool_factor = pool_factor

        # Pre-compute lengths
        self.src_lengths = [len(s) for s in dataset.src]
        self.tgt_lengths = [len(t) for t in dataset.tgt]

        # Create batches
        self._batches = self._create_batches()

    def _create_batches(self):
        """Create batches grouped by approximate sequence length."""
        # Sort indices by max(src_len, tgt_len) for efficient batching
        indices = list(range(len(self.dataset)))
        indices.sort(key=lambda i: (max(self.src_lengths[i], self.tgt_lengths[i]),
                                     self.src_lengths[i]))

        # Create batches from sorted indices
        batches = []
        current_batch = []
        current_max_src = 0
        current_max_tgt = 0

        for idx in indices:
            src_len = self.src_lengths[idx]
            tgt_len = self.tgt_lengths[idx]

            # Check if adding this example would exceed budget
            new_max_src = max(current_max_src, src_len)
            new_max_tgt = max(current_max_tgt, tgt_len)
            new_batch_size = len(current_batch) + 1

            # Estimated tokens = batch_size * max_length (due to padding)
            est_src_tokens = new_batch_size * new_max_src
            est_tgt_tokens = new_batch_size * new_max_tgt

            if current_batch and (est_src_tokens > self.tokens_per_batch or
                                   est_tgt_tokens > self.tokens_per_batch):
                batches.append(current_batch)
                current_batch = [idx]
                current_max_src = src_len
                current_max_tgt = tgt_len
            else:
                current_batch.append(idx)
                current_max_src = new_max_src
                current_max_tgt = new_max_tgt

        if current_batch:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        batches = self._batches.copy()
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self._batches)


def collate_fn(batch_items):
    """
    Collate a list of (src, tgt) pairs into padded tensors.

    Returns dict with:
        src: (batch_size, max_src_len) LongTensor
        tgt: (batch_size, max_tgt_len) LongTensor
        src_mask: (batch_size, 1, max_src_len) BoolTensor (True where not padded)
        tgt_mask: (batch_size, 1, max_tgt_len) BoolTensor (True where not padded)
        ntokens: int, total non-pad tokens in target
    """
    src_seqs = [item["src"] for item in batch_items]
    tgt_seqs = [item["tgt"] for item in batch_items]

    max_src_len = max(len(s) for s in src_seqs)
    max_tgt_len = max(len(t) for t in tgt_seqs)

    batch_size = len(src_seqs)

    src_tensor = torch.full((batch_size, max_src_len), PAD_ID, dtype=torch.long)
    tgt_tensor = torch.full((batch_size, max_tgt_len), PAD_ID, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_seqs, tgt_seqs)):
        src_tensor[i, :len(src)] = torch.tensor(src, dtype=torch.long)
        tgt_tensor[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)

    src_mask = (src_tensor != PAD_ID).unsqueeze(1)  # (B, 1, S)
    tgt_mask = (tgt_tensor != PAD_ID).unsqueeze(1)  # (B, 1, T)

    ntokens = tgt_mask.sum().item()

    return {
        "src": src_tensor,
        "tgt": tgt_tensor,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "ntokens": ntokens,
    }


def create_dataloader(dataset, tokens_per_batch=25000, shuffle=True, num_workers=0):
    """
    Create a DataLoader with token-based batching.

    Args:
        dataset: TranslationDataset instance
        tokens_per_batch: Target ~25,000 source and ~25,000 target tokens per batch
        shuffle: Whether to shuffle batch order each epoch
        num_workers: Number of data loading workers

    Returns:
        DataLoader instance
    """
    sampler = TokenBatchSampler(
        dataset,
        tokens_per_batch=tokens_per_batch,
        shuffle=shuffle,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return loader


# Convenience: load from default paths
def load_en_de(outputs_dir=None, tokens_per_batch=25000, shuffle=True):
    """Load EN-DE dataset and return dataloader."""
    if outputs_dir is None:
        outputs_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(outputs_dir, "en_de_tokenized.pt")
    dataset = TranslationDataset(data_path)
    return create_dataloader(dataset, tokens_per_batch=tokens_per_batch, shuffle=shuffle)


def load_en_fr(outputs_dir=None, tokens_per_batch=25000, shuffle=True):
    """Load EN-FR dataset and return dataloader."""
    if outputs_dir is None:
        outputs_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(outputs_dir, "en_fr_tokenized.pt")
    dataset = TranslationDataset(data_path)
    return create_dataloader(dataset, tokens_per_batch=tokens_per_batch, shuffle=shuffle)
