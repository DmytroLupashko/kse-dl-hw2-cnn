"""Run folder round trip: `save_run` then `load_runs`. Run: `python -m unittest discover -s tests`."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml

from core.training import runs
from core.training.config import CLASSES


class TestRuns(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runs, "PROCESSED", Path(tmp)):
            config = {"experiment": "cnn-test", "name": "net", "data": {"img_size": 64}}
            path = Path(tmp) / "net.yaml"
            path.write_text(yaml.safe_dump(config))
            oofs = {}

            for fold in (1, 0):
                out = runs.run_dir("cnn-test", "net", fold)
                out.mkdir(parents=True)
                (out / "config.yaml").write_text(yaml.safe_dump(config))
                oofs[fold] = pd.DataFrame(
                    [[fold, *[1] * len(CLASSES), *[0.5] * len(CLASSES)]],
                    columns=["fold", *CLASSES, *runs.PROB_COLS],
                    index=[10 + fold],
                )
                runs.save_run(out, f"run{fold}", oofs[fold])

            # unfinished fold (no `run.json`) and full run ignored
            runs.run_dir("cnn-test", "net", 2).mkdir()
            full = runs.run_dir("cnn-test", "net", None)
            full.mkdir()
            runs.save_run(full, "full")

            loaded, oof, run_ids, dirs = runs.load_runs(path)

        self.assertEqual(loaded, config)
        self.assertEqual(run_ids, {0: "run0", 1: "run1"})
        self.assertEqual(list(dirs), [0, 1])
        pd.testing.assert_frame_equal(oof, pd.concat([oofs[0], oofs[1]]))

    def test_no_finished_fold(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runs, "PROCESSED", Path(tmp)):
            path = Path(tmp) / "net.yaml"
            path.write_text(yaml.safe_dump({"experiment": "cnn-test", "name": "net"}))

            with self.assertRaises(FileNotFoundError):
                runs.load_runs(path)


if __name__ == "__main__":
    unittest.main()
