import os
import pkgutil
import tempfile
from functools import lru_cache
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import (
    SmilesDataset,
    calculate_frechet_distance,
    load_imported_model,
    todevice,
)


@lru_cache(maxsize=1)
def load_ref_model(model_path: Optional[str] = None):
    """Loads chemnet model

    Args:
        model_path (str | None, optional): Path to model file. Defaults to None.

    Returns:
        Chemnet as torch model
    """

    if model_path is None:
        chemnet_model_filename = "ChemNet_v0.13_pretrained.pt"
        model_bytes = pkgutil.get_data("fcd", chemnet_model_filename)

        tmpdir = tempfile.TemporaryDirectory()
        model_path = os.path.join(tmpdir.name, chemnet_model_filename)
        with open(model_path, "wb") as f:
            f.write(model_bytes)

    model_config = torch.load(model_path)
    model = load_imported_model(model_config)
    model.eval()
    return model


def get_predictions(
    model: nn.Module,
    smiles_list: List[str],
    batch_size: int = 128,
    n_jobs: int = 1,
    device: str = "cpu",
) -> np.ndarray:
    """Calculate Chemnet activations

    Args:
        model (nn.Module): Chemnet model
        smiles_list (List[str]): List of smiles to process
        batch_size (int, optional): Which batch size to use for inference. Defaults to 128.
        n_jobs (int, optional): How many jobs to use for preprocessing. Defaults to 1.
        device (str, optional): On which device the chemnet model is run. Defaults to "cpu".

    Returns:
        np.ndarray: The activation for the input list
    """
    if len(smiles_list) == 0:
        return np.zeros((0, 512))

    dataloader = DataLoader(
        SmilesDataset(smiles_list), batch_size=batch_size, num_workers=n_jobs
    )
    with todevice(model, device), torch.no_grad():
        chemnet_activations = []
        for batch in dataloader:
            chemnet_activations.append(
                model(batch.transpose(1, 2).float().to(device))
                .to("cpu")
                .detach()
                .numpy()
            )
    return np.row_stack(chemnet_activations)


def get_fcd(smiles1: List[str], smiles2: List[str], model: nn.Module = None) -> float:
    """Calculate FCD between two sets of Smiles

    Args:
        smiles1 (List[str]): First set of smiles
        smiles2 (List[str]): Second set of smiles
        model (nn.Module, optional): The model to use. Loads default model if None.

    Returns:
        float: The FCD score
    """
    if model is None:
        model = load_ref_model()

    act1 = get_predictions(model, smiles1)
    act2 = get_predictions(model, smiles2)

    mu1 = np.mean(act1, axis=0)
    sigma1 = np.cov(act1.T)

    mu2 = np.mean(act2, axis=0)
    sigma2 = np.cov(act2.T)

    fcd_score = calculate_frechet_distance(
        mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2
    )

    return fcd_score
