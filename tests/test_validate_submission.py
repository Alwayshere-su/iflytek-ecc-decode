from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "validate_submission.py"
BITS = [f"bit_{i}" for i in range(64)]
HEADER = ",".join(["id", *BITS])


def run_validator(csv_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(csv_path), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


class ValidateSubmissionTests(unittest.TestCase):
    def test_zero_codeword_passes_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            path.write_text(
                HEADER + "\n"
                + "0," + ",".join(["0"] * 64) + "\n"
                + "1," + ",".join(["0"] * 64) + "\n",
                encoding="utf-8",
            )
            result = run_validator(path, "--expect-rows", "2")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("RESULT: PASS", result.stdout)

    def test_rejects_non_binary_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            row = ["0"] * 64
            row[0] = "2"
            path.write_text(HEADER + "\n0," + ",".join(row) + "\n", encoding="utf-8")
            result = run_validator(path, "--expect-rows", "1", "--skip-parity")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-binary", result.stdout)

    def test_rejects_wrong_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            path.write_text("id,bit_0\n0,0\n", encoding="utf-8")
            result = run_validator(path, "--skip-parity")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("column mismatch", result.stdout)

    def test_rejects_wrong_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            path.write_text(HEADER + "\n0," + ",".join(["0"] * 64) + "\n", encoding="utf-8")
            result = run_validator(path, "--expect-rows", "2")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("row count", result.stdout)

    def test_reference_submission_is_valid(self) -> None:
        gz_path = ROOT / "results" / "submission_o4.csv.gz"
        if not gz_path.exists():
            self.skipTest("reference submission is not available")
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "submission_o4.csv"
            with gzip.open(gz_path, "rb") as source, csv_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            result = run_validator(csv_path, "--expect-rows", "100000", "--skip-parity")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("RESULT: PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
