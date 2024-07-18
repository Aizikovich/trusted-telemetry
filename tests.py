import unittest
from utils import update_cols, get_dfs, create_idxs, scale_df, load_csvs, features_extraction
import pandas as pd


class TestStringMethods(unittest.TestCase):

    def test_steps_in_load_csvs(self):
        ues, cells = load_csvs()
        for j, ue in enumerate(ues):
            steps = len(ue['step'].unique())
            print(ue.shape[0] / 50, steps)
            self.assertEqual(ue.shape[0] / 50, steps, msg=f"ues index {j} has problem with steps")
        print("----")
        for j, cell in enumerate(cells):
            steps = len(cell['step'].unique())
            print(cell.shape[0] / 6, steps)
            self.assertEqual(cell.shape[0] / 6, steps, msg=f"cells index {j} has problem with steps")

    def test_dataframes(self):
        ues, cells = load_csvs()

        for ue in ues:
            self.assertEqual({50}, set(ue.groupby('step')['ue-id'].count()), msg=f"some steps doesnt have 50 ues rows")
            # test if there are 50 unique ues every step
            for s in ue['step'].unique():
                self.assertEqual(50, len(ue[ue['step'] == s]['ue-id'].unique()),
                                 msg=f"some steps doesnt have 50 unique ues fail at step {s}")

    def test_get_dfs(self):
        ues, cells = load_csvs()
        dfs = get_dfs(ues, cells)

        for df in dfs:
            steps = len(df['step'].unique())
            print(df.shape[0] / 6, steps)
            self.assertEqual(df.shape[0], steps, msg=f"df has problem with steps")


if __name__ == '__main__':
    unittest.main()
