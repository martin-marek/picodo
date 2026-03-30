import os

import numpy as np


def load_ds(seed, ds_path, seq_len, batch_size, n_tokens_valid, n_tokens_train=None):

    # get dataset size
    print('getting dataset size...')
    ds_path = os.path.expanduser(ds_path)
    tokens = np.memmap(ds_path, dtype=np.uint16, mode='r')
    n_tokens_dataset = len(tokens)

    # if n_tokens_train is None, use full dataset
    if n_tokens_train is not None: assert n_tokens_train + n_tokens_valid <= n_tokens_dataset
    if n_tokens_train is None: n_tokens_train = n_tokens_dataset - n_tokens_valid

    # get num. of train. and valid. sequences / batches
    n_seq_train = n_tokens_train // seq_len
    n_seq_valid = n_tokens_valid // seq_len
    n_batch_train = n_seq_train // batch_size
    n_batch_valid = n_seq_valid // batch_size
    n_seq = (n_batch_train + n_batch_valid) * batch_size

    # memmap contiguous sequences
    print('reading data...')
    data = np.memmap(ds_path, dtype=np.uint16, shape=[n_seq, seq_len], mode='r')

    # shuffle sequences, then group them into batches
    print('shuffling data...')
    rng = np.random.default_rng(seed)
    batch_indices = rng.permutation(n_seq).astype(np.int32).reshape(-1, batch_size)

    # split data
    idx_train = batch_indices[:n_batch_train]
    idx_valid = batch_indices[n_batch_train:]
    
    return data, idx_train, idx_valid
