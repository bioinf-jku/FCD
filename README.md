# Fréchet ChemNet Distance

The new wave of successful generative models in machine learning has increased
the interest in deep learning driven de novo drug design. However, assessing
the performance of such generative models is notoriously difficult. Metrics that
are typically used to assess the performance of such generative models are the
percentage of chemically valid molecules or the similarity to real molecules in
terms of particular descriptors, such as the partition coefficient (logP) or druglike-
ness. However, method comparison is difficult because of the inconsistent use of
evaluation metrics, the necessity for multiple metrics, and the fact that some of
these measures can easily be tricked by simple rule-based systems. We propose a
novel distance measure between two sets of molecules, called Fréchet ChemNet
distance (FCD), that can be used as an evaluation metric for generative models. The
FCD is similar to a recently established performance metric for comparing image
generation methods, the Fréchet Inception Distance (FID). Whereas the FID uses
one of the hidden layers of InceptionNet, the FCD utilizes the penultimate layer
of a deep neural network called “ChemNet”, which was trained to predict drug
activities. Thus, the FCD metric takes into account chemically and biologically
relevant information about molecules, and also measures the diversity of the set
via the distribution of generated molecules. The FCD’s advantage over previous
metrics is that it can detect if generated molecules are a) diverse and have similar
b) chemical and c) biological properties as real molecules. We further provide an
easy-to-use implementation that only requires the SMILES representation of the
generated molecules as input to calculate the FCD.

## Version 1.1 changes
- Got rid of unneeded imports
- `load_ref_model` doesn't need an argument any more to load a model.
- `canonical` and `canonical_smiles` now return `None` for invalid Smiles.
- Added `get_fcd` as a quick way to get a score