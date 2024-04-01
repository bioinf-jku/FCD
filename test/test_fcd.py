import numpy as np
import pytest
from pytest import approx

from fcd import get_fcd
from fcd.utils import SmilesDataset, get_one_hot


class TestFCD:
    def test_random_smiles_cpu(self):
        smiles_list1 = ["CNOHF", "NOHFCl", "OHFClP", "HFClPB"]
        smiles_list2 = ["ISi#()", "Si#()+", "#()+-", "()+-1"]
        target = 8.8086
        fcd = get_fcd(smiles_list1, smiles_list2, device="cpu")
        assert fcd == approx(target, abs=1e-2)

    def test_random_smiles_gpu(self):
        # Skip test if CUDA is not available
        # CUDA comp is less consistent than CPU
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        smiles_list1 = ["CNOHF", "NOHFCl", "OHFClP", "HFClPB"]
        smiles_list2 = ["ISi#()", "Si#()+", "#()+-", "()+-1"]
        target = 8.8086
        fcd = get_fcd(smiles_list1, smiles_list2, device="cuda")
        assert fcd == approx(target, abs=1e-2)

    def test_random_smiles_cpu_2(self):
        smiles_list1 = [
            "COc1cccc(NC(=O)Cc2coc3ccc(OC)cc23)c1",
            "Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1",
        ]
        smiles_list2 = [
            "Oc1ccccc1-c1cccc2cnccc12",
            "Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1",
        ]
        target = 47.773486382119444
        fcd = get_fcd(smiles_list1, smiles_list2, device="cpu")

        assert fcd == approx(target, abs=1e-3)

    def test_random_smiles_gpu_2(self):
        # Skip test if CUDA is not available
        # CUDA comp is less consistent than CPU
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        smiles_list1 = [
            "COc1cccc(NC(=O)Cc2coc3ccc(OC)cc23)c1",
            "Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1",
        ]
        smiles_list2 = [
            "Oc1ccccc1-c1cccc2cnccc12",
            "Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1",
        ]
        target = 47.773486382119444
        fcd = get_fcd(smiles_list1, smiles_list2)

        assert fcd == approx(target, abs=1e-3)

    def test_one_hot(self):
        inputs = [
            "O=C([C@H]1CC[C@@H]2[C@@H]",
            "CCN(CC)S(=O)(=O)CCC[C@H](",
            r"COc1ccc(\C=C\2/C[N+](C)(C",
            "COc1cc(CC(=O)N2CCN(CC2)c3",
            "CN(CC#Cc1cc(ccc1C)ClC2(O)CC",
            "CYACCyyCLCl",
        ]
        # fmt: off
        outputs = [
            ((26, 35), [2, 25, 0, 13, 26, 0, 28, 3, 27, 17, 0, 0, 26, 0, 28, 28, 3, 27, 18, 26, 0, 28, 28, 3, 27, 34]),
            ((26, 35), [0, 0, 1, 13, 0, 0, 14, 9, 13, 25, 2, 14, 13, 25, 2, 14, 0, 0, 0, 26, 0, 28, 3, 27, 13, 34]),
            ((26, 35), [0, 2, 29, 17, 29, 29, 29, 13, 33, 0, 25, 0, 33, 18, 33, 0, 26, 1, 15, 27, 13, 0, 14, 13, 0, 34]),
            ((26, 35), [0, 2, 29, 17, 29, 29, 13, 0, 0, 13, 25, 2, 14, 1, 18, 0, 0, 1, 13, 0, 0, 18, 14, 29, 19, 34]),
            ((28, 35), [0, 1, 13, 0, 0, 12, 0, 29, 17, 29, 29, 13, 29, 29, 29, 17, 0, 14, 5, 0, 18, 13, 2, 14, 0, 0, 34]),
            ((12, 35), [0, 33, 33, 0, 0, 33, 33, 0, 33, 5, 34]),
        ]
        # fmt: on

        for inp, (correct_shape, correct_entries) in zip(inputs, outputs):
            one_hot = get_one_hot(inp)
            shape = one_hot.shape
            entries = np.where(one_hot)[1].tolist()
            assert shape == correct_shape
            assert entries == correct_entries

            # assert that no duplicate ones and no missing entries. Trailing zero vectors are allowed.
            non_zero_idx = np.where(one_hot)[0]
            assert np.all(non_zero_idx == np.arange(len(non_zero_idx)))

    def test_one_hot_padding(self):
        smiles = "CNOHFCCCCCCCC"
        pad_len = 5
        with pytest.raises(AssertionError):
            one_hot = get_one_hot(smiles, pad_len=pad_len)


class TestUtils:
    def test_canonicalize(self):
        from fcd.utils import canonical_smiles

        smiles = [
            "O=C(OCC(=O)NC=1C=CC=C(OC)C1)COC=2C=CC=CC2C",
            "O=C(NC1CCN(C(=O)CC=2C=CC=CC2O)CC1)CC=3C=CC=C(F)C3F",
            "O=C1CCN(CC2=CN(C=3C=NC=CC32)C)CC1(C4=CC=C(OC)C=C4)C5=CC=C(OC)C=C5",
        ]
        can_smiles_correct = [
            "COc1cccc(NC(=O)COC(=O)COc2ccccc2C)c1",
            "O=C(Cc1cccc(F)c1F)NC1CCN(C(=O)Cc2ccccc2O)CC1",
            "COc1ccc(C2(c3ccc(OC)cc3)CN(Cc3cn(C)c4cnccc34)CCC2=O)cc1",
        ]

        can_smiles = canonical_smiles(smiles)

        assert can_smiles == can_smiles_correct


class TestSmilesDataset:
    def test_dataset_okay(self):
        smiles = ["CNOHF", "NOHFCl", "OHFClP", "HFClPB"]
        smiles_dataset = SmilesDataset(smiles)
        assert len(smiles_dataset) == len(smiles)
        assert smiles_dataset.pad_len == 350

    def test_smiles_too_long(self):
        """Check if warning is raised when smiles are too long for default pad_length"""

        smiles = ["CNOHF" * 100, "NOHFCl", "OHFClP", "HFClPB"]
        with pytest.warns(UserWarning):
            smiles_dataset = SmilesDataset(smiles)

        assert len(smiles_dataset) == len(smiles)
        assert smiles_dataset.pad_len == 501  # plus one for the end token

    def test_smiles_one_off(self):
        smiles = ["CCCCC"]
        with pytest.warns(UserWarning):
            smiles_dataset = SmilesDataset(smiles, pad_len=len(smiles[0]) + 1)  # plus one for the end token

        assert isinstance(smiles_dataset[0], np.ndarray)

    def test_custom_pad_length(self):
        """Check if custom pad_length is used and warning is issued"""
        smiles = ["CNOHF", "NOHFCl", "OHFClP", "HFClPB"]
        with pytest.warns(UserWarning):
            smiles_dataset = SmilesDataset(smiles, pad_len=20)

        assert len(smiles_dataset) == len(smiles)
        assert smiles_dataset.pad_len == 20
