import unittest

import numpy as np
import pandas as pd

from analysis import HypoCriteria, bp_points, bp_raw


class BloodPressureTests(unittest.TestCase):
    def test_read_only_input_detects_measurements_without_mutating_source(self):
        values = np.array(
            [
                [0, 140, 80, 70],
                [60, 140, 80, 70],
                [120, -1, -1, -1],
                [180, 120, 70, 75],
                [240, 120, 70, 75],
                [300, 85, 55, 80],
                [360, 85, 55, 80],
            ],
            dtype=float,
        )
        values.flags.writeable = False
        source = pd.DataFrame(
            values, columns=["treat_sec", "sbp", "dbp", "pulse"], copy=False
        )
        original = source.copy(deep=True)

        result = bp_points(source, HypoCriteria())

        self.assertEqual(result["treat_sec"].tolist(), [0, 180, 300])
        self.assertEqual(result["sbp"].tolist(), [140, 120, 85])
        self.assertEqual(result["seq"].tolist(), [0, 1, 2])
        self.assertEqual(result["hypo"].tolist(), [False, True, True])
        np.testing.assert_allclose(result["d_sbp_base"], [0, -20, -55])
        pd.testing.assert_frame_equal(source, original)

    def test_all_invalid_measurements_are_excluded(self):
        source = pd.DataFrame(
            {"treat_sec": [0.0, 60.0], "sbp": [-1.0, np.nan],
             "dbp": [-1.0, np.nan], "pulse": [-1.0, np.nan]}
        )
        original = source.copy(deep=True)

        result = bp_raw(source)

        self.assertTrue(result.empty)
        self.assertIn("seq", result.columns)
        pd.testing.assert_frame_equal(source, original)


if __name__ == "__main__":
    unittest.main()
