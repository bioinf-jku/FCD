# Fréchet ChemNet Distance

Code for the paper "Fréchet ChemNet Distance: A Metric for Generative Models for Molecules in Drug Discovery"
[JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00234) /
[ArXiv](https://arxiv.org/abs/1803.09518)


## Installation
You can install the FCD using
```
pip install fcd
```

rdkit is best install via conda
```
conda install rdkit -c rdkit
```

# Requirements
```
numpy
tensorflow>1.8
keras>2.1
scipy
```
For the effect of versions on results see `tests/test_results.csv`.
Using the current versions of tensorflow (2.1.0) and keras (2.3.1) results differ from previous versions but
are probably negligible.


## Version 1.1 changes
- Got rid of unneeded imports
- `load_ref_model` doesn't need an argument any more to load a model.
- `canonical` and `canonical_smiles` now return `None` for invalid smiles.
- Added `get_fcd` as a quick way to get a the fcd score from two lists of smiles.