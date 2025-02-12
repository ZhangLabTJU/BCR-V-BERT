import os
import json
import math
import torch
import itertools
import tempfile
import logging
import vbert
import numpy as np
import pandas as pd
from itertools import zip_longest
from typing import *
from transformers import BertTokenizer
from math import floor



AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
PAD = "$"
MASK = "."
UNK = "?"
SEP = "|"
CLS = "*"
AMINO_ACIDS_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AMINO_ACIDS_WITH_ALL_ADDITIONAL = AMINO_ACIDS + PAD + MASK + UNK + SEP + CLS
AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX = {
    aa: i for i, aa in enumerate(AMINO_ACIDS_WITH_ALL_ADDITIONAL)
}

def load_blosum(
    fname: str = "utils/blosum62.json"
) -> pd.DataFrame:
    """Return the blosum matrix as a dataframe"""
    with open(fname) as source:
        d = json.load(source)
        retval = pd.DataFrame(d)
    retval = pd.DataFrame(0, index=list(AMINO_ACIDS), columns=list(AMINO_ACIDS))
    for x, y in itertools.product(retval.index, retval.columns):
        retval.loc[x, y] = d[x][y]
    return retval

def get_aa_bert_tokenizer(
    max_len: int, d=AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX
) -> BertTokenizer:
    """
    Tokenizer for amino acid sequences. Not *exactly* the same as BertTokenizer
    but mimics its behavior, encoding start with CLS and ending with SEP
    >>> get_aa_bert_tokenizer(10).encode(insert_whitespace("RKDES"))
    [25, 0, 2, 3, 4, 5, 24]
    """
    with tempfile.TemporaryDirectory() as tempdir:
        vocab_fname = write_vocab(d, os.path.join(tempdir, "vocab.txt"))
        tok = BertTokenizer(
            vocab_fname,
            do_lower_case=False,
            do_basic_tokenize=True,
            tokenize_chinese_chars=False,
            pad_token=PAD,
            mask_token=MASK,
            unk_token=UNK,
            sep_token=SEP,
            cls_token=CLS,
            model_max_len=max_len,
            padding_side="right",
        )
    return tok

def write_vocab(vocab: Iterable[str], fname: str) -> str:
    """
    Write the vocabulary to the fname, one entry per line
    Mostly for compatibility with transformer BertTokenizer
    """
    with open(fname, "w") as sink:
        for v in vocab:
            sink.write(v + "\n")
    return fname

def pad_or_trunc_sequence(seq: str, l: int, right_align: bool = False, pad=PAD) -> str:
    """
    Pad the given sequence to the given length
    >>> pad_or_trunc_sequence("RKDES", 8, right_align=False)
    'RKDES$$$'
    >>> pad_or_trunc_sequence("RKDES", 8, right_align=True)
    '$$$RKDES'
    >>> pad_or_trunc_sequence("RKDESRKRKR", 3, right_align=False)
    'RKD'
    >>> pad_or_trunc_sequence("RKDESRRK", 3, right_align=True)
    'RRK'
    """
    delta = len(seq) - l
    if len(seq) > l:
        if right_align:
            retval = seq[delta:]
        else:
            retval = seq[:-delta]
    elif len(seq) < l:
        insert = pad * np.abs(delta)
        if right_align:
            retval = insert + seq
        else:
            retval = seq + insert
    else:
        retval = seq
    assert len(retval) == l, f"Got mismatched lengths: {len(retval)} {l}"
    return retval

def insert_whitespace(seq: str) -> str:
    """
    Return the sequence of characters with whitespace after each char
    >>> insert_whitespace("RKDES")
    'R K D E S'
    """
    return " ".join(list(seq))

def adheres_to_vocab(s: str, vocab: str = AMINO_ACIDS) -> bool:
    """
    Returns whether a given string contains only characters from vocab
    >>> adheres_to_vocab("RKDES")
    True
    >>> adheres_to_vocab(AMINO_ACIDS + AMINO_ACIDS)
    True
    """
    return set(s).issubset(set(vocab))

def insert_whitespace(seq: str) -> str:
    """
    Return the sequence of characters with whitespace after each char
    >>> insert_whitespace("RKDES")
    'R K D E S'
    """
    return " ".join(list(seq))

def is_whitespaced(seq: str) -> bool:
    """
    Return whether the sequence has whitespace inserted
    >>> is_whitespaced("R K D E S")
    True
    >>> is_whitespaced("RKDES")
    False
    >>> is_whitespaced("R K D ES")
    False
    >>> is_whitespaced("R")
    True
    >>> is_whitespaced("RK")
    False
    >>> is_whitespaced("R K")
    True
    """
    tok = list(seq)
    spaces = [t for t in tok if t.isspace()]
    if len(spaces) == floor(len(seq) / 2):
        return True
    return False

def one_hot(seq: str, alphabet: Optional[str] = AMINO_ACIDS) -> np.ndarray:
    """
    One-hot encode the input string. Since pytorch convolutions expect
    input of (batch, channel, length), we return shape (channel, length)
    When one hot encoding, we ignore the pad characters, encoding them as
    a vector of 0's instead
    """
    if not seq:
        assert alphabet
        return np.zeros((len(alphabet), 1), dtype=np.float32)
    if not alphabet:
        alphabet = dedup(seq)
        logging.info(f"No alphabet given, assuming alphabet of: {alphabet}")
    seq_arr = np.array(list(seq))
    # This implementation naturally ignores the pad character if not provided
    # in the alphabet
    retval = np.stack([seq_arr == char for char in alphabet]).astype(float).T
    assert len(retval) == len(seq), f"Mismatched lengths: {len(seq)} {retval.shape}"
    return retval.astype(np.float32).T

def min_power_greater_than(
    value: SupportsFloat, base: SupportsFloat = 2
) -> SupportsFloat:
    """
    Return the lowest power of the base that exceeds the given value
    >>> min_power_greater_than(3, 4)
    4.0
    >>> min_power_greater_than(48, 2)
    64.0
    """
    p = math.ceil(math.log(value, base))
    return math.pow(base, p)

def get_transformer_embeddings(
    model_dir: str,
    tok,
    seqs: Iterable[str],
    vgene: Iterable[int],
    seq_pair: Optional[Iterable[str]] = None,
    *,
    layers: List[int] = [-1],
    method: Literal["mean", "max", "attn_mean", "cls", "pool"] = "mean",
    batch_size: int = 256,
    device: int = 0,
    max_len = 66
) -> np.ndarray:
    """
    Get the embeddings for the given sequences from the given layers
    Layers should be given as negative integers
    Returns a matrix of num_seqs x (hidden_dim * len(layers))
    Methods:
    - cls:  value of initial CLS token
    - mean: average of sequence length, excluding initial CLS token
    - max:  maximum over sequence length, excluding initial CLS token
    - attn_mean: mean over sequenced weighted by attention, excluding initial CLS token
    - pool: pooling layer
    If multiple layers are given, applies the given method to each layers
    and concatenate across layers
    """
    device = get_device(device)
    seqs = [s if is_whitespaced(s) else insert_whitespace(s) for s in seqs]
    try:
        # tok = ft.get_pretrained_bert_tokenizer(model_dir)
        tok = tok
    except OSError:
        logging.warning("Could not load saved tokenizer, loading fresh instance")
        # tok = ft.get_aa_bert_tokenizer(max_len)
        tok = tok
    model = vbert.VBertModel.from_pretrained(model_dir, add_pooling_layer=method == "pool").to(
        device
    )# model = vbert.my_BertModel.from_pretrained

    chunks = chunkify(seqs, batch_size)
    chunk_vgene = chunkify(vgene, batch_size)
    # This defaults to [None] to zip correctly
    chunks_pair = [None]
    if seq_pair is not None:
        assert len(seq_pair) == len(seqs)
        chunks_pair = chunkify(
            [s if is_whitespaced(s) else insert_whitespace(s) for s in seq_pair],
            batch_size,
        )
    # For single input, we get (list of seq, None) items
    # for a duo input, we get (list of seq1, list of seq2)
    chunks_zipped = list(zip_longest(chunks, chunks_pair))
    embeddings = []
    with torch.no_grad():
        for idx in range(len(chunks_zipped)):
            encoded = tok(
                *chunks_zipped[idx], padding="max_length", max_length=max_len, return_tensors="pt"
            )
            vgene_idx = torch.from_numpy(np.array(chunk_vgene[idx]))
            vgene_idx = torch.unsqueeze(vgene_idx,1).expand(encoded["input_ids"].size())
            # manually calculated mask lengths
            # temp = [sum([len(p.split()) for p in pair]) + 3 for pair in zip(*seq_chunk)]
            # input_mask = encoded["attention_mask"].numpy()
            encoded["vgene_ids"] = vgene_idx
            encoded = {k: v.to(device) for k, v in encoded.items()}
            # encoded contains input attention mask of (batch, seq_len)
            x = model.forward(
                **encoded, output_hidden_states=True, output_attentions=True
            )
            if method == "pool":
                embeddings.append(x.pooler_output.cpu().numpy().astype(np.float64))
                continue
            # x.hidden_states contains hidden states, num_hidden_layers + 1
            # Each hidden state is (batch, seq_len, hidden_size)
            # x.hidden_states[-1] == x.last_hidden_state
            # x.attentions contains attention, num_hidden_layers
            # Each attention is (batch, attn_heads, seq_len, seq_len)

            for i in range(len(chunks_zipped[idx][0])):
                e = []
                for l in layers:
                    # Select the l-th hidden layer for the i-th example
                    h = (
                        x.hidden_states[l][i].cpu().numpy().astype(np.float64)
                    )  # seq_len, hidden
                    # initial 'cls' token
                    if method == "cls":
                        e.append(h[0])
                        continue
                    # Consider rest of sequence
                    if chunks_zipped[idx][1] is None:
                        seq_len = len(chunks_zipped[idx][0][i].split())  # 'R K D E S' = 5
                    else:
                        seq_len = (
                            len(chunks_zipped[idx][0][i].split())
                            + len(chunks_zipped[idx][1][i].split())
                            + 1  # For the sep token
                        )
                    seq_hidden = h[1 : 1 + seq_len]  # seq_len * hidden
                    assert len(seq_hidden.shape) == 2
                    if method == "mean":
                        e.append(seq_hidden.mean(axis=0))
                    elif method == "max":
                        e.append(seq_hidden.max(axis=0))
                    elif method == "attn_mean":
                        # (attn_heads, seq_len, seq_len)
                        # columns past seq_len + 2 are all 0
                        # summation over last seq_len dim = 1 (as expected after softmax)
                        attn = x.attentions[l][i, :, :, : seq_len + 2]
                        # print(attn.shape)
                        print(attn.sum(axis=-1))
                        raise NotImplementedError
                    else:
                        raise ValueError(f"Unrecognized method: {method}")
                e = np.hstack(e)
                assert len(e.shape) == 1
                embeddings.append(e)
    if len(embeddings[0].shape) == 1:
        embeddings = np.stack(embeddings)
    else:
        embeddings = np.vstack(embeddings)
    del x
    del model
    torch.cuda.empty_cache()
    return embeddings

def get_transformer_all_embeddings(
    model_dir: str,
    tok,
    seqs: Iterable[str],
    vgene: Iterable[int],
    seq_pair: Optional[Iterable[str]] = None,
    *,
    layers: List[int] = [-1],
    method: Literal["mean", "max", "attn_mean", "cls", "pool"] = "mean",
    batch_size: int = 256,
    device: int = 0,
    max_len = 66
) -> np.ndarray:
    """
    Get the embeddings for the given sequences from the given layers
    Layers should be given as negative integers
    Returns a matrix of num_seqs x (hidden_dim * len(layers))
    Methods:
    - cls:  value of initial CLS token
    - mean: average of sequence length, excluding initial CLS token
    - max:  maximum over sequence length, excluding initial CLS token
    - attn_mean: mean over sequenced weighted by attention, excluding initial CLS token
    - pool: pooling layer
    If multiple layers are given, applies the given method to each layers
    and concatenate across layers
    """
    device = get_device(device)
    seqs = [s if is_whitespaced(s) else insert_whitespace(s) for s in seqs]
    try:
        # tok = ft.get_pretrained_bert_tokenizer(model_dir)
        tok = tok
    except OSError:
        logging.warning("Could not load saved tokenizer, loading fresh instance")
        # tok = ft.get_aa_bert_tokenizer(max_len)
        tok = tok
    model = vbert.VBertModel.from_pretrained(model_dir, add_pooling_layer=method == "pool").to(
        device
    )# model = vbert.my_BertModel.from_pretrained

    chunks = chunkify(seqs, batch_size)
    chunk_vgene = chunkify(vgene, batch_size)
    # This defaults to [None] to zip correctly
    chunks_pair = [None]
    if seq_pair is not None:
        assert len(seq_pair) == len(seqs)
        chunks_pair = chunkify(
            [s if is_whitespaced(s) else insert_whitespace(s) for s in seq_pair],
            batch_size,
        )
    # For single input, we get (list of seq, None) items
    # for a duo input, we get (list of seq1, list of seq2)
    chunks_zipped = list(zip_longest(chunks, chunks_pair))
    hidden_embeddings = []
    with torch.no_grad():
        for idx in range(len(chunks_zipped)):
            encoded = tok(
                *chunks_zipped[idx], padding="max_length", max_length=max_len, return_tensors="pt"
            )
            vgene_idx = torch.from_numpy(np.array(chunk_vgene[idx]))
            vgene_idx = torch.unsqueeze(vgene_idx,1).expand(encoded["input_ids"].size())
            # manually calculated mask lengths
            # temp = [sum([len(p.split()) for p in pair]) + 3 for pair in zip(*seq_chunk)]
            # input_mask = encoded["attention_mask"].numpy()
            encoded["vgene_ids"] = vgene_idx
            encoded = {k: v.to(device) for k, v in encoded.items()}
            # encoded contains input attention mask of (batch, seq_len)
            x = model.forward(
                **encoded, output_hidden_states=True, output_attentions=True
            )
            if method == "pool":
                hidden_embeddings.append(x.pooler_output.cpu().numpy().astype(np.float64))
                continue
            # x.hidden_states contains hidden states, num_hidden_layers + 1
            # Each hidden state is (batch, seq_len, hidden_size)
            # x.hidden_states[-1] == x.last_hidden_state
            # x.attentions contains attention, num_hidden_layers
            # Each attention is (batch, attn_heads, seq_len, seq_len)

            for i in range(len(chunks_zipped[idx][0])):
                for l in layers:
                    # Select the l-th hidden layer for the i-th example
                    h = (
                        x.hidden_states[l][i].cpu().numpy().astype(np.float64)
                    )  # seq_len, hidden
                    seq_hidden = h
                hidden_embeddings.append(seq_hidden)
    hidden_embeddings = np.array(hidden_embeddings)
    del x
    del model
    torch.cuda.empty_cache()
    return hidden_embeddings

def get_device(i: Optional[int] = None) -> str:
    """
    Returns the i-th GPU if GPU is available, else CPU
    A negative value or a float will default to CPU
    """
    # if torch.cuda.is_available() and i is not None and isinstance(i, int) and i >= 0:
    #     devices = list(range(torch.cuda.device_count()))
    #     device_idx = devices[i]
    #     torch.cuda.set_device(device_idx)
    #     d = torch.device(f"cuda:{device_idx}")
    #     torch.cuda.set_device(d)
    # else:
    #     logging.warn("Defaulting to CPU")
    #     d = torch.device("cpu")
    d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return d

def chunkify(x: Sequence[Any], chunk_size: int = 128):
    """
    Split list into chunks of given size
    >>> chunkify([1, 2, 3, 4, 5, 6, 7], 3)
    [[1, 2, 3], [4, 5, 6], [7]]
    >>> chunkify([(1, 10), (2, 20), (3, 30), (4, 40)], 2)
    [[(1, 10), (2, 20)], [(3, 30), (4, 40)]]
    """
    retval = [x[i : i + chunk_size] for i in range(0, len(x), chunk_size)]
    return retval

def dedup(x: Iterable[Any]) -> List[Any]:
    """
    Dedup the given iterable, preserving order of occurrence
    >>> dedup([1, 2, 0, 1, 3, 2])
    [1, 2, 0, 3]
    >>> dedup(dedup([1, 2, 0, 1, 3, 2]))
    [1, 2, 0, 3]
    """
    # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    # Python 3.7 and above guarantee that dict is insertion ordered
    # sets do NOT do this, so list(set(x)) will lose order information
    return list(dict.fromkeys(x))
