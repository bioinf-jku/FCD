import os
import sys

import keras
import tensorflow
from fcd import get_fcd, load_ref_model





# Don't use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def dummyfcd():
    one_hot = [
        'C', 'N', 'O', 'H', 'F', 'Cl', 'P', 'B', 'Br', 'S', 'I', 'Si',
        '#', '(', ')', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '=', '[', ']', '@',
        'c', 'n', 'o', 's', 'X', '.']

    # get sum dummy sequences
    s1 = []
    s2 = []
    for i in range(4):
        s1.append(''.join(one_hot[i:i + 5]))
        s2.append(''.join(one_hot[i + 10:i + 15]))

    return get_fcd(s1, s2)


def test_dummyfcd():
    fcd_score = dummyfcd()
    assert abs(fcd_score - 8.808623423861544) < 1e-4


if __name__ == '__main__':
    fcd_score = dummyfcd

    with open(os.path.join(os.path.dirname(__file__), 'dummyfcd_results.csv'), 'a') as f:
        f.write(', '.join([str(fcd_score), str(
            tensorflow.__version__), str(keras.__version__)]) + '\n')
