# Fréchet ChemNet Distance

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