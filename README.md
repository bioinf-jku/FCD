# Fréchet ChemNet Distance
![PyPI](https://img.shields.io/pypi/v/fcd)
![Tests (master)](https://github.com/bioinf-jku/fcd/actions/workflows/test_master.yml/badge.svg?branch=dev)
![Tests (dev)](https://github.com/bioinf-jku/fcd/actions/workflows/test_dev.yml/badge.svg?branch=dev)
![PyPI - Downloads](https://img.shields.io/pypi/dm/fcd)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/bioinf-jku/fcd)
![GitHub release date](https://img.shields.io/github/release-date/bioinf-jku/fcd)
![GitHub](https://img.shields.io/github/license/bioinf-jku/fcd)


Code for the paper "Fréchet ChemNet Distance: A Metric for Generative Models for Molecules in Drug Discovery"
[JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00234) /
[ArXiv](https://arxiv.org/abs/1803.09518)


## Installation
You can install the FCD using
```
pip install fcd
```

# Requirements
```
numpy
torch
scipy
rdkit
```

# Updates
## Version 1.1 changes
- Got rid of unneeded imports
- `load_ref_model` doesn't need an argument any more to load a model.
- `canonical` and `canonical_smiles` now return `None` for invalid smiles.
- Added `get_fcd` as a quick way to get a the fcd score from two lists of smiles.

## Version 1.2 changes
- Ported the package to pytorch with the help of https://github.com/insilicomedicine/fcd_torch
- pytorch allows a lighter package and is more popular than Tensorflow which saves an additional install