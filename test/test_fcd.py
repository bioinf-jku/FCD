import unittest

import numpy as np

from fcd import get_fcd
from fcd.utils import get_one_hot


class TestFCD(unittest.TestCase):
    def test_random_smiles(self):
        smiles_list1 = ["CNOHF", "NOHFCl", "OHFClP", "HFClPB"]
        smiles_list2 = ["ISi#()", "Si#()+", "#()+-", "()+-1"]
        target = 8.8086
        self.assertAlmostEqual(
            get_fcd(smiles_list1, smiles_list2), target, 3, f"Should be {target}"
        )

    def test_fcd_torch(self):
        smiles_list1 = [
            "COc1cccc(NC(=O)Cc2coc3ccc(OC)cc23)c1",
            "Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1",
        ]
        smiles_list2 = [
            "Oc1ccccc1-c1cccc2cnccc12",
            "Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1",
        ]
        target = 47.773486382119444
        self.assertAlmostEqual(
            get_fcd(smiles_list1, smiles_list2), target, 3, f"Should be {target}"
        )

    def test_one_hot(self):
        inputs = [
            "O=C([C@H]1CC[C@@H]2[C@@H]",
            "CCN(CC)S(=O)(=O)CCC[C@H](",
            "COc1ccc(\C=C\2/C[N+](C)(C",
            "COc1cc(CC(=O)N2CCN(CC2)c3",
            "CN(CC#Cc1cc(ccc1C)ClC2(O)CC",
            "CYACCyyCLCl",
        ]
        # fmt: off
        outputs = [
            ((26, 35), [2, 25, 0, 13, 26, 0, 28, 3, 27, 17, 0, 0, 26, 0, 28, 28, 3, 27, 18, 26, 0, 28, 28, 3, 27, 34]),
            ((26, 35), [0, 0, 1, 13, 0, 0, 14, 9, 13, 25, 2, 14, 13, 25, 2, 14, 0, 0, 0, 26, 0, 28, 3, 27, 13, 34]),
            ((25, 35), [0, 2, 29, 17, 29, 29, 29, 13, 33, 0, 25, 0, 33, 33, 0, 26, 1, 15, 27, 13, 0, 14, 13, 0, 34]),
            ((26, 35), [0, 2, 29, 17, 29, 29, 13, 0, 0, 13, 25, 2, 14, 1, 18, 0, 0, 1, 13, 0, 0, 18, 14, 29, 19, 34]),
            ((28, 35), [0, 1, 13, 0, 0, 12, 0, 29, 17, 29, 29, 13, 29, 29, 29, 17, 0, 14, 5, 0, 18, 13, 2, 14, 0, 0, 34]),
            ((12, 35), [0, 33, 33, 0, 0, 33, 33, 0, 33, 5, 34]),
        ]
        # fmt: on

        for inp, (correct_shape, correct_entries) in zip(inputs, outputs):
            one_hot = get_one_hot(inp)
            shape = one_hot.shape
            entries = np.where(one_hot)[1].tolist()
            self.assertEqual(shape, correct_shape)
            self.assertEqual(entries, correct_entries)

            # assert that no duplicate ones and no missing entries. Trailing zero vectors are allowed.
            non_zero_idx = np.where(one_hot)[0]
            assert np.all(non_zero_idx == np.arange(len(non_zero_idx)))


if __name__ == "__main__":
    unittest.main()
