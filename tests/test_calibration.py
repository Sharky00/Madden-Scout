import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calibration import align_plays, values_match


class CalibrationAlignmentTests(unittest.TestCase):
    def test_ocr_whitespace_does_not_count_as_call_mismatch(self) -> None:
        self.assertTrue(values_match("offensive_play_call", "HB Stretch", "HBSTRETCH"))

    def test_inserted_observation_does_not_shift_later_plays(self) -> None:
        expected = [
            {"play_number": 0, "down": 2, "distance": 9, "yard_line": 36},
            {"play_number": 1, "down": 3, "distance": 1, "yard_line": 44},
            {"play_number": 2, "down": 4, "distance": 2, "yard_line": 43},
        ]
        observed = [
            {"play_number": 0, "down": 2, "distance": 9, "yard_line": 36},
            {"play_number": 1, "down": 1, "distance": 10, "yard_line": 3},
            {"play_number": 2, "down": 3, "distance": 1, "yard_line": 44},
            {"play_number": 3, "down": 4, "distance": 2, "yard_line": 43},
        ]

        aligned = align_plays(expected, observed)

        self.assertEqual(0, aligned[0]["play_number"])
        self.assertEqual(2, aligned[1]["play_number"])
        self.assertEqual(3, aligned[2]["play_number"])


if __name__ == "__main__":
    unittest.main()
