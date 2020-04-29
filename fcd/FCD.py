#!/usr/bin/env python3
''' Defines the functions necessary for calculating the Frechet ChemNet
Distance (FCD) to evalulate generative models for molecules.

The FCD metric calculates the distance between two distributions of molecules.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by the generative
model.

The FCD is calculated by assuming that X_1 and X_2 are the activations of
the preulitmate layer of the CHEMNET for generated samples and real world
samples respectivly.
'''

from __future__ import absolute_import, division, print_function

import os
import pkgutil
import tempfile
import warnings
from functools import lru_cache
from multiprocessing import Pool

import keras.backend as K
import numpy as np
from keras.models import load_model
from rdkit import Chem
from scipy import linalg

warnings.filterwarnings('ignore')


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1:    The mean of the activations of preultimate layer of the
               CHEMNET ( like returned by the function 'get_predictions')
               for generated samples.
    -- mu2:    The mean of the activations of preultimate layer of the
               CHEMNET ( like returned by the function 'get_predictions')
               for real samples.
    -- sigma1: The covariance matrix of the activations of preultimate layer of the
               CHEMNET ( like returned by the function 'get_predictions')
               for generated samples.
    -- sigma2: The covariance matrix of the activations of preultimate layer of the
               CHEMNET ( like returned by the function 'get_predictions')
               for real samples.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

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


def build_masked_loss(loss_function, mask_value):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function


def masked_accuracy(y_true, y_pred):
    a = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), K.floatx()))
    c = K.sum(K.cast(K.not_equal(y_true, 0.5), K.floatx()))
    acc = (a) / c
    return acc


def get_one_hot(smiles, pad_len=-1):
    one_hot = [
        'C', 'N', 'O', 'H', 'F', 'Cl', 'P', 'B', 'Br', 'S', 'I', 'Si',
        '#', '(', ')', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '=', '[', ']', '@',
        'c', 'n', 'o', 's', 'X', '.']
    smiles = smiles + '.'
    if pad_len < 0:
        vec = np.zeros((len(smiles), len(one_hot)))
    else:
        vec = np.zeros((pad_len, len(one_hot)))
    cont = True
    j = 0
    i = 0
    while cont:
        if smiles[i + 1] in ['r', 'i', 'l']:
            sym = smiles[i:i + 2]
            i += 2
        else:
            sym = smiles[i]
            i += 1
        if sym in one_hot:
            vec[j, one_hot.index(sym)] = 1
        else:
            vec[j, one_hot.index('X')] = 1
        j += 1
        if smiles[i] == '.' or j >= (pad_len - 1) and pad_len > 0:
            vec[j, one_hot.index('.')] = 1
            cont = False
    return (vec)


def myGenerator_predict(smilesList, batch_size=128, pad_len=350):
    while 1:
        N = len(smilesList)
        nn = pad_len
        idxSamples = np.arange(N)

        for j in range(int(np.ceil(N / batch_size))):
            idx = idxSamples[j * batch_size: min((j + 1) * batch_size, N)]

            x = []
            for i in range(0, len(idx)):
                currentSmiles = smilesList[idx[i]]
                smiEnc = get_one_hot(currentSmiles, pad_len=nn)
                x.append(smiEnc)

            x = np.asarray(x) / 35
            yield x


@lru_cache(maxsize=4)
def load_ref_model(model_file=None):
    """Loads the Chemnet model. If called without arguments it will use the
    model in the package. In case you want to use a different one provide the path

    Args:
        model_file: Path to model. (default=None)
    """

    if model_file is None:
        chemnet_model_filename = 'ChemNet_v0.13_pretrained.h5'
        model_bytes = pkgutil.get_data('fcd', chemnet_model_filename)

        tmpdir = tempfile.TemporaryDirectory()
        model_file = os.path.join(tmpdir.name, chemnet_model_filename)

        with open(model_file, 'wb') as f:
            f.write(model_bytes)

    masked_loss_function = build_masked_loss(K.binary_crossentropy, 0.5)
    model = load_model(
        model_file,
        custom_objects={
            'masked_loss_function': masked_loss_function,
            'masked_accuracy': masked_accuracy})
    model.pop()
    model.pop()
    return model


def get_predictions(model, gen_mol):
    gen_mol_act = model.predict_generator(
        myGenerator_predict(gen_mol, batch_size=128),
        steps=np.ceil(len(gen_mol) / 128))
    return gen_mol_act


def canonical(smi):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        return None


def canonical_smiles(smiles, njobs=32):
    with Pool(njobs) as pool:
        return pool.map(canonical, smiles)


def get_fcd(smiles1, smiles2, model=None):
    if model is None:
        model = load_ref_model()

    act1 = get_predictions(model, smiles1)
    act2 = get_predictions(model, smiles2)

    mu1 = np.mean(act1, axis=0)
    sigma1 = np.cov(act1.T)

    mu2 = np.mean(act2, axis=0)
    sigma2 = np.cov(act2.T)

    fcd_score = calculate_frechet_distance(
        mu1=mu1,
        mu2=mu2,
        sigma1=sigma1,
        sigma2=sigma2)

    return fcd_score
