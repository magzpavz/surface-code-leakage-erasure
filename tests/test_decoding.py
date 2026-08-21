import unittest

from surface_code_leakage_erasure.decoding import ErasureDecoder


class BacktrackValidityTest(unittest.TestCase):
    def test_conflict_witness_does_not_prune_valid_sibling(self):
        required = {"c", "d", "f"}
        dem_table = [
            [{"a", "c"}, {"d", "e"}],
            [{"d", "e", "f"}, {"b", "c", "e"}],
        ]

        # Choosing candidate A from both rows covers every required edge.
        self.assertTrue(required <= dem_table[0][0] | dem_table[1][0])

        decoder = ErasureDecoder.__new__(ErasureDecoder)
        valid, witness = decoder.backtrack(dem_table, uncovered=required)

        self.assertTrue(valid, f"valid combination was pruned; witness={witness}")
        self.assertEqual(witness, set())

    def test_failed_sibling_witnesses_are_lifted_to_their_parent(self):
        required = {"c", "f"}
        dem_table = [[{"f"}, {"c"}]]

        decoder = ErasureDecoder.__new__(ErasureDecoder)
        valid, witness = decoder.backtrack(dem_table, uncovered=required)

        self.assertFalse(valid)
        self.assertEqual(witness, required)


if __name__ == "__main__":
    unittest.main()
