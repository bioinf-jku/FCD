import unittest

from fcd import get_fcd


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


if __name__ == "__main__":
    unittest.main()
