import unittest

from IDM_pred.io import get_connectivity_PCs, get_df, subject_sets


class TestIO(unittest.TestCase):
    def test_subject_sets(self):
        for key, val in subject_sets.items():
            s = sorted(val)
            self.assertEqual(val, s)
        self.assertEqual(len(subject_sets['s876']), 876)
        self.assertEqual(len(subject_sets['s888']), 888)

    def test_get_df(self):
        df = get_df()
        df = df.loc[subject_sets['s888']]
        self.assertEqual(df.shape[0], 888)


if __name__ == '__main__':
    unittest.main()
