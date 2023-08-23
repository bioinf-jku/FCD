import re
from contextlib import contextmanager
from multiprocessing import Pool
from typing import List

import numpy as np
import torch
from rdkit import Chem
from scipy import linalg
from torch import nn
from torch.utils.data import Dataset

from .torch_layers import IndexTensor, IndexTuple, Reverse, SamePadding1d, Transpose

# fmt: off
__vocab = ["C","N","O","H","F","Cl","P","B","Br","S","I","Si","#","(",")","+","-","1","2","3","4","5","6","7","8","=","[","]","@","c","n","o","s","X","."]
# fmt: on
__vocab_c2i = {k: i for i, k in enumerate(__vocab)}
__unk = __vocab_c2i["X"]


sorted_vocab = __vocab[:]
sorted_vocab.sort()  # sort alphabetically
sorted_vocab.sort(key=len, reverse=True)  # sort by length for regex
FULL_REGEX = "|".join(
    "(%s)" % re.escape(base_symbol) for base_symbol in sorted_vocab
)  # Tries to match longer tokens first.
FULL_REGEX += "|."  # Handle unkown characters


def tokenize(smiles: str) -> List[str]:
    """Tokenizes the given smiles string. Needed for multi-character tokens like 'Cl'

    Args:
        smiles (str): Input molecule as Smiles

    Returns:
        List[str]: List of tokens
    """
    tok_smile = [mo.group() for mo in re.finditer(FULL_REGEX, smiles)]
    assert "".join(tok_smile) == smiles
    return tok_smile


def get_one_hot(smiles: str, pad_len: int = -1) -> np.ndarray:
    """Generate one-hot representation of a Smiles string.

    Args:
        smiles (str): Input molecule as Smiles
        pad_len (int, optional): Whether or not to pad to a given size. Defaults to -1.

    Returns:
        np.ndarray: Array containing the one-hot encoded Smiles
    """
    smiles = smiles + "."

    # initialize array
    array_length = len(smiles) if pad_len < 0 else pad_len
    vocab_size = len(__vocab)
    one_hot = np.zeros((array_length, vocab_size))

    tokens = tokenize(smiles)
    numeric = [__vocab_c2i.get(token, __unk) for token in tokens]

    for pos, num in enumerate(numeric):
        one_hot[pos, num] = 1

    return one_hot


def load_imported_model(keras_config):
    activations = {
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    layers = []
    conv = True
    for layer_type, config in keras_config:
        state_dict, kwargs, other_info = config
        if layer_type == "Conv1d":
            assert conv, "Can't have conv layers after LSTM"
            if other_info["padding"] == "same":
                layers.append(SamePadding1d(kwargs["kernel_size"], kwargs["stride"]))
            layer = nn.Conv1d(**kwargs)
            layer.load_state_dict(state_dict)
            layers.append(layer)
            activation = other_info["activation"]
            layers.append(activations[activation]())
        elif layer_type == "LSTM":
            if conv:
                conv = False
                layers.append(Transpose())
            layer = nn.LSTM(**kwargs)
            layer.load_state_dict(state_dict)
            if other_info["reverse"]:
                layers.append(Reverse())
            layers.append(layer)
            layers.append(IndexTuple(0))
            if other_info["last"]:
                layers.append(IndexTensor(-1, 1))
        else:
            raise ValueError("Unknown layer type")
    return nn.Sequential(*layers)


class SmilesDataset(Dataset):
    __PAD_LEN = 350

    def __init__(self, smiles_list):
        super().__init__()
        self.smiles_list = smiles_list

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        features = get_one_hot(smiles, 350)
        return features / features.shape[1]

    def __len__(self):
        return len(self.smiles_list)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1:    The mean of the activations of preultimate layer of the
               CHEMNET (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2:    The mean of the activations of preultimate layer of the
               CHEMNET (like returned by the function 'get_predictions')
               for real samples.
    -- sigma1: The covariance matrix of the activations of preultimate layer
               of the CHEMNET (like returned by the function 'get_predictions')
               for generated samples.
    -- sigma2: The covariance matrix of the activations of preultimate layer
               of the CHEMNET (like returned by the function 'get_predictions')
               for real samples.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


@contextmanager
def todevice(model, device):
    model.to(device)
    yield
    model.to("cpu")
    torch.cuda.empty_cache()


def canonical(smi):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        return None


def canonical_smiles(smiles, njobs=32):
    with Pool(njobs) as pool:
        return pool.map(canonical, smiles)
