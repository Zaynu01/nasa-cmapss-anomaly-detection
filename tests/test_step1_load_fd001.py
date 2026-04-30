from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cmapss_loader import COLUMNS, load_cmapss_table, load_rul_values, validate_fd001_dataset


class TestStep1LoadFd001(unittest.TestCase):
    def test_fd001_train_loads_with_expected_shape(self) -> None:
        train = load_cmapss_table(PROJECT_ROOT / "nasa" / "train_FD001.txt")

        self.assertEqual(train.columns, COLUMNS)
        self.assertEqual(train.row_count, 20631)
        self.assertEqual(train.column_count, 26)
        self.assertEqual(train.engine_count, 100)

    def test_fd001_test_and_rul_counts_match(self) -> None:
        test = load_cmapss_table(PROJECT_ROOT / "nasa" / "test_FD001.txt")
        rul = load_rul_values(PROJECT_ROOT / "nasa" / "RUL_FD001.txt")

        self.assertEqual(test.row_count, 13096)
        self.assertEqual(test.engine_count, 100)
        self.assertEqual(len(rul), test.engine_count)

    def test_fd001_full_validation_summary(self) -> None:
        summary = validate_fd001_dataset(PROJECT_ROOT / "nasa")

        self.assertEqual(summary["train_rows"], 20631)
        self.assertEqual(summary["test_rows"], 13096)
        self.assertEqual(summary["train_engines"], 100)
        self.assertEqual(summary["test_engines"], 100)
        self.assertEqual(summary["rul_values"], 100)


if __name__ == "__main__":
    unittest.main()
