import json
import os
import sys
import tempfile
import unittest
from datetime import date
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import video_intake as vi


class SituationPlayTests(unittest.TestCase):
    def test_final_csv_name_uses_resulting_score_and_completion_date(self) -> None:
        plays = [
            {
                "home_team": "TEN",
                "home_score": 23,
                "away_team": "NYJ",
                "away_score": 27,
                "end_home_score": 29,
                "end_away_score": 27,
            }
        ]

        filename = vi.final_football_csv_name(plays, date(2026, 8, 19))

        self.assertEqual("TEN 29 - 27 NYJ 2026-08-19.csv", filename)

    def test_final_csv_name_falls_back_to_last_displayed_score(self) -> None:
        plays = [
            {
                "home_team": "TEN",
                "home_score": 29,
                "away_team": "NYJ",
                "away_score": 27,
            }
        ]

        filename = vi.final_football_csv_name(plays, date(2026, 8, 19))

        self.assertEqual("TEN 29 - 27 NYJ 2026-08-19.csv", filename)

    def test_final_csv_path_uses_flat_game_data_directory(self) -> None:
        plays = [
            {
                "home_team": "TEN",
                "home_score": 29,
                "away_team": "NYJ",
                "away_score": 27,
            }
        ]

        path = vi.final_football_csv_path("output", plays, date(2026, 8, 19))

        self.assertEqual(
            os.path.join(
                "output",
                "Game_Data",
                "TEN 29 - 27 NYJ 2026-08-19.csv",
            ),
            path,
        )

    def test_team_csv_paths_include_both_scoreboard_teams(self) -> None:
        plays = [{"home_team": "TEN", "away_team": "NYJ"}]
        filename = "TEN 29 - 27 NYJ 2026-08-19.csv"

        paths = vi.team_football_csv_paths("output", plays, filename)

        self.assertEqual(
            [
                os.path.join("output", "Team_Data", "NYJ", filename),
                os.path.join("output", "Team_Data", "TEN", filename),
            ],
            paths,
        )

    def test_team_database_initializes_all_reference_teams(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            directories = vi.initialize_team_database(output_dir)

            self.assertEqual(32, len(directories))
            self.assertTrue(all(os.path.isdir(path) for path in directories))

    def test_impossible_down_regression_repairs_to_fourth_down(self) -> None:
        summaries = [
            {
                "down": 3,
                "distance": 0,
                "down_distance": "3 & inches",
                "special_state": None,
            },
            {"down": 2, "distance": 2, "down_distance": "2 & 2", "special_state": None},
            {
                "down": 1,
                "distance": 10,
                "down_distance": "1 & 10",
                "special_state": None,
            },
        ]

        vi.repair_down_progression(summaries)

        self.assertEqual(4, summaries[1]["down"])
        self.assertEqual("4 & 2", summaries[1]["down_distance"])

    def test_unanchored_result_banner_attaches_to_next_special_state(self) -> None:
        base = {
            "quarter": 1,
            "game_clock": "5:57",
            "away_team": "NYJ",
            "home_team": "TEN",
            "away_score": 0,
            "home_score": 0,
            "field_side": "opponent",
            "stream_side": "offense",
            "personnel": None,
            "formation": None,
            "prev_off_call": None,
            "prev_def_call": None,
            "previous_play_banner_shown": False,
        }
        states = [
            {**base, "down": 2, "distance": 9, "special_state": None, "yard_line": 42},
            {
                **base,
                "down": None,
                "distance": None,
                "special_state": None,
                "yard_line": None,
                "prev_off_call": "HB BASE",
                "prev_def_call": "SS DOG 3",
                "previous_play_banner_shown": True,
            },
            {
                **base,
                "down": None,
                "distance": None,
                "special_state": "extra_point",
                "yard_line": 15,
                "home_score": 6,
            },
            {
                **base,
                "down": None,
                "distance": None,
                "special_state": "extra_point",
                "yard_line": 15,
                "home_score": 6,
            },
        ]
        records = [
            vi.FrameRecord(index, float(index), str(index)) for index in range(4)
        ]

        with patch.object(vi, "extract_frame_state", side_effect=states):
            plays = vi.build_situation_plays(
                records,
                vi.Region(0, 0, 1, 1),
                vi.Region(0, 0, 1, 1),
            )

        self.assertEqual("HB BASE", plays[0]["offensive_play_call"])
        self.assertIsNone(plays[0]["defensive_play_call"])
        self.assertEqual("SS DOG 3", plays[0]["defensive_play_call_ocr"])
        self.assertEqual("unverified", plays[0]["defensive_play_call_validation"])
        self.assertEqual("touchdown", plays[0]["result"])

    def test_fuzzy_call_consensus_merges_minor_ocr_variants(self) -> None:
        consensus = vi.fuzzy_text_consensus(
            ["HBSTRETCH", "HB STRETCH", "HB STRETCH", "HE STRETCH"]
        )

        self.assertEqual("HB STRETCH", consensus)

    def test_play_call_validation_uses_catalog_canonical_name(self) -> None:
        catalog = {
            "offense": ["H Mtn Flood H-Sneak", "Mtn Flood Hb Sneak"],
            "defense": ["Roll 23 Cloud", "Roll 28 Cloud"],
        }

        offensive, offensive_status = vi.validate_play_call(
            "H MIN FLOOD HSNEAK", "offense", catalog
        )
        defensive, defensive_status = vi.validate_play_call(
            "ROLL 23 CHOUP", "defense", catalog
        )

        self.assertEqual("H MTN FLOOD H-SNEAK", offensive)
        self.assertEqual("verified_fuzzy", offensive_status)
        self.assertEqual("ROLL 23 CLOUD", defensive)
        self.assertEqual("verified_fuzzy", defensive_status)

    def test_play_call_validation_rejects_ambiguous_ocr(self) -> None:
        catalog = {"offense": ["Mesh", "Mesh Spot"], "defense": []}

        play_call, status = vi.validate_play_call("ME", "offense", catalog)

        self.assertIsNone(play_call)
        self.assertEqual("unverified", status)

    def test_short_exact_catalog_name_is_verified(self) -> None:
        catalog = {"offense": ["Duo", "Y Out"], "defense": [], "plays": []}

        self.assertEqual(
            ("DUO", "verified"),
            vi.validate_play_call("DUO", "offense", catalog),
        )
        self.assertEqual(
            ("Y OUT", "verified"),
            vi.validate_play_call("YOUT", "offense", catalog),
        )

    def test_offense_fuzzy_match_can_be_constrained_by_formation(self) -> None:
        catalog = {
            "offense": ["Pa Deep Read", "Pa Deep Out"],
            "defense": [],
            "plays": [
                {
                    "side": "offense",
                    "formation_family": "Gun",
                    "formation": "Trips TE",
                    "play_name": "Pa Deep Read",
                }
            ],
        }

        call, status = vi.validate_play_call(
            "PA DEEP REAO", "offense", catalog, "Gun - Trips TE"
        )

        self.assertEqual("PA DEEP READ", call)
        self.assertEqual("verified_fuzzy", status)

    def test_catalog_play_type_uses_consistent_name_metadata(self) -> None:
        catalog = {
            "plays": [
                {
                    "side": "offense",
                    "formation_family": "Gun",
                    "formation": "Trips TE",
                    "play_name": "Wide Trap",
                    "play_type": "run",
                },
                {
                    "side": "offense",
                    "formation_family": "Singleback",
                    "formation": "Ace",
                    "play_name": "Wide Trap",
                    "play_type": "run",
                },
            ]
        }

        self.assertEqual("run", vi.catalog_play_type("WIDE TRAP", None, catalog))

    def test_catalog_play_type_requires_formation_for_conflicting_name(self) -> None:
        catalog = {
            "plays": [
                {
                    "side": "offense",
                    "formation_family": "Gun",
                    "formation": "Normal",
                    "play_name": "HB Base",
                    "play_type": "run",
                },
                {
                    "side": "offense",
                    "formation_family": "Singleback",
                    "formation": "Ace",
                    "play_name": "HB Base",
                    "play_type": "pass",
                },
            ]
        }

        self.assertIsNone(vi.catalog_play_type("HB BASE", None, catalog))
        self.assertEqual(
            "run", vi.catalog_play_type("HB BASE", "Gun - Normal", catalog)
        )

    def test_catalog_unique_formation_requires_one_catalog_formation(self) -> None:
        catalog = {
            "plays": [
                {
                    "side": "offense",
                    "formation_family": "Gun",
                    "formation": "Deuce Close",
                    "play_name": "Bench Dig Curl",
                }
            ]
        }

        self.assertEqual(
            "Gun - Deuce Close",
            vi.catalog_unique_formation("BENCH DIG CURL", catalog),
        )

    def test_catalog_unique_formation_rejects_ambiguous_play(self) -> None:
        catalog = {
            "plays": [
                {
                    "side": "offense",
                    "formation_family": "Gun",
                    "formation": "Trips TE",
                    "play_name": "Verticals",
                },
                {
                    "side": "offense",
                    "formation_family": "Singleback",
                    "formation": "Ace",
                    "play_name": "Verticals",
                },
            ]
        }

        self.assertIsNone(vi.catalog_unique_formation("VERTICALS", catalog))

    def test_catalog_consensus_requires_repeated_support(self) -> None:
        catalog = {"offense": ["HB Stretch"], "defense": [], "plays": []}

        self.assertEqual(
            "HB STRETCH",
            vi.catalog_candidate_consensus(
                ["HB STRETCH", "HB STRETCH", "HE STRETCH"],
                "offense",
                catalog,
            ),
        )
        self.assertIsNone(
            vi.catalog_candidate_consensus(["HB STRETCH"], "offense", catalog)
        )

    def test_catalog_consensus_does_not_invent_repeated_non_catalog_text(self) -> None:
        catalog = {
            "offense": ["Smash Dagger"],
            "defense": [],
            "plays": [],
        }

        self.assertIsNone(
            vi.catalog_candidate_consensus(
                ["SMASH DRAG", "SMASH DRAG", "SMASH DRAG"],
                "offense",
                catalog,
            )
        )

    def test_catalog_unavailable_does_not_publish_raw_ocr(self) -> None:
        play_call, status = vi.validate_play_call(
            "HB STRETCH", "offense", {"offense": [], "defense": []}
        )

        self.assertIsNone(play_call)
        self.assertEqual("catalog_unavailable", status)

    def test_output_audit_rejects_non_catalog_claim(self) -> None:
        with (
            patch.object(
                vi,
                "load_play_catalog",
                return_value={"offense": ["HB Stretch"], "defense": ["Cover 3 Sky"]},
            ),
            self.assertRaisesRegex(RuntimeError, "Non-catalog play calls"),
        ):
            vi.assert_catalog_valid_calls(
                [{"offensive_play_call": "OCR GARBAGE", "play_type": "run"}]
            )

    def test_output_audit_accepts_catalog_and_special_team_calls(self) -> None:
        with patch.object(
            vi,
            "load_play_catalog",
            return_value={"offense": ["HB Stretch"], "defense": ["Cover 3 Sky"]},
        ):
            vi.assert_catalog_valid_calls(
                [
                    {
                        "offensive_play_call": "HB STRETCH",
                        "defensive_play_call": "COVER 3 SKY",
                        "play_type": "run",
                    },
                    {"offensive_play_call": "PAT Kick", "play_type": "extra_point"},
                ]
            )

    def test_catalog_validation_report_separates_claims_from_unverified_ocr(
        self,
    ) -> None:
        plays = [
            {
                "offensive_play_call": "HB STRETCH",
                "defensive_play_call": None,
                "offensive_play_call_validation": "verified",
                "defensive_play_call_validation": "unverified",
                "defensive_play_call_ocr": "OCR GARBAGE",
            }
        ]
        with (
            patch.object(
                vi,
                "load_play_catalog",
                return_value={"source": "catalog.json"},
            ),
            patch.object(vi, "open", mock_open()) as mocked_open,
        ):
            vi.write_catalog_validation_report(plays, "output")

        report = json.loads(
            "".join(call.args[0] for call in mocked_open().write.call_args_list)
        )
        self.assertTrue(report["all_claimed_calls_in_catalog"])
        self.assertEqual(1, report["claimed_call_cells"])
        self.assertEqual(1, report["unverified_ocr_cells"])

    def test_missing_previous_play_banner_is_explicit(self) -> None:
        calls = vi.resolve_summary_calls(
            {"_banner_shown": False, "_prev_off": None, "_prev_def": None},
            {"offense": [], "defense": []},
        )

        self.assertEqual(vi.CALL_BANNER_NOT_SHOWN, calls[0])
        self.assertEqual(vi.CALL_BANNER_NOT_SHOWN, calls[1])
        self.assertEqual(("banner_not_shown", "banner_not_shown"), calls[2:])

    def test_unverified_boundary_panel_is_treated_as_banner_not_shown(self) -> None:
        previous = {
            "start_timestamp_seconds": 1.0,
            "quarter": 1,
            "game_clock": "6:27",
            "away_team": "AAA",
            "away_score": 0,
            "home_team": "BBB",
            "home_score": 0,
            "down": 4,
            "distance": 2,
            "down_distance": "4 & 2",
            "yard_line": 43,
            "_stream_side": "defense",
            "possession_team": "AAA",
            "field_side": "own",
            "_personnel": None,
            "_formation": None,
            "_banner_shown": True,
            "_prev_off": None,
            "_prev_def": None,
            "special_state": None,
        }
        current = {
            "start_timestamp_seconds": 5.0,
            "down": 1,
            "distance": 10,
            "down_distance": "1 & 10",
            "yard_line": 43,
            "special_state": None,
            "possession_team": "BBB",
            "_banner_shown": True,
            "_prev_off": "OCR GARBAGE",
            "_prev_def": "MORE GARBAGE",
            "away_score": 0,
            "home_score": 0,
        }

        play = vi.pair_to_play(previous, current)

        self.assertFalse(play["previous_play_banner_shown"])
        self.assertEqual("banner_not_shown", play["offensive_play_call_validation"])
        self.assertEqual(vi.CALL_BANNER_NOT_SHOWN, play["offensive_play_call"])
        self.assertEqual("OCR GARBAGE", play["offensive_play_call_ocr"])

    def test_verified_boundary_call_is_preserved(self) -> None:
        catalog = {"offense": ["Double Cross"], "defense": [], "plays": []}
        previous = {
            "start_timestamp_seconds": 1.0,
            "quarter": 1,
            "game_clock": "2:27",
            "away_team": "AAA",
            "away_score": 0,
            "home_team": "BBB",
            "home_score": 0,
            "down": 4,
            "distance": 3,
            "down_distance": "4 & 3",
            "yard_line": 11,
            "_stream_side": "defense",
            "possession_team": "AAA",
            "field_side": "own",
            "_personnel": None,
            "_formation": None,
            "_banner_shown": False,
            "_prev_off": None,
            "_prev_def": None,
            "special_state": None,
        }
        current = {
            "start_timestamp_seconds": 5.0,
            "down": 1,
            "distance": 10,
            "down_distance": "1 & 10",
            "yard_line": 45,
            "special_state": None,
            "possession_team": "BBB",
            "_banner_shown": True,
            "_prev_off": "DOUBLE CROSS",
            "_prev_def": None,
            "away_score": 0,
            "home_score": 0,
        }

        with patch.object(vi, "load_play_catalog", return_value=catalog):
            play = vi.pair_to_play(previous, current)

        self.assertTrue(play["previous_play_banner_shown"])
        self.assertEqual("DOUBLE CROSS", play["offensive_play_call"])

    def test_pair_preserves_ocr_and_records_fuzzy_catalog_match(self) -> None:
        catalog = {"offense": ["HB Stretch"], "defense": ["Cover 3 Sky"], "plays": []}
        prev = {
            "start_timestamp_seconds": 1.0,
            "quarter": 1,
            "game_clock": "7:00",
            "away_team": "AAA",
            "away_score": 0,
            "home_team": "BBB",
            "home_score": 0,
            "down": 1,
            "distance": 10,
            "down_distance": "1 & 10",
            "yard_line": 25,
            "_stream_side": "offense",
            "possession_team": "AAA",
            "field_side": "own",
            "_personnel": None,
            "_formation": None,
            "_banner_shown": False,
            "_prev_off": None,
            "_prev_def": None,
            "special_state": None,
        }
        cur = {
            "start_timestamp_seconds": 5.0,
            "down": 2,
            "distance": 8,
            "down_distance": "2 & 8",
            "yard_line": 27,
            "special_state": None,
            "possession_team": "AAA",
            "_banner_shown": True,
            "_prev_off": "HB STRETCH",
            "_prev_def": "COVER 3 SKV",
            "away_score": 0,
            "home_score": 0,
        }

        with patch.object(vi, "load_play_catalog", return_value=catalog):
            play = vi.pair_to_play(prev, cur)

        self.assertEqual("COVER 3 SKY", play["defensive_play_call"])
        self.assertEqual("COVER 3 SKV", play["defensive_play_call_ocr"])
        self.assertEqual("COVER 3 SKY", play["defensive_play_call_catalog_match"])
        self.assertEqual("verified_fuzzy", play["defensive_play_call_validation"])

    def test_score_stabilization_rejects_single_frame_noise(self) -> None:
        states = [
            {"away_score": value, "home_score": 0} for value in (0, 0, 7, 0, 0, 6, 6, 6)
        ]

        vi.stabilize_frame_states(states)

        self.assertEqual(
            [0, 0, 0, 0, 0, 6, 6, 6], [state["away_score"] for state in states]
        )

    def test_one_point_score_growth_requires_extra_point_evidence(self) -> None:
        states = [
            {"away_score": score, "special_state": None} for score in (0, 0, 1, 1, 3, 3)
        ]

        vi.stabilize_score_field(states, "away_score")

        self.assertNotIn(1, [state["away_score"] for state in states])

    def test_one_point_score_growth_is_kept_near_extra_point(self) -> None:
        states = [
            {"away_score": 0, "special_state": "extra_point"},
            {"away_score": 1, "special_state": "kickoff"},
            {"away_score": 1, "special_state": None},
        ]

        vi.stabilize_score_field(states, "away_score")

        self.assertEqual(1, states[-1]["away_score"])

    def test_situation_stabilization_rejects_isolated_yard_noise(self) -> None:
        states = [
            {"down": 3, "distance": 3, "yard_line": 41, "special_state": None},
            {"down": 4, "distance": 3, "yard_line": 1, "special_state": None},
            {"down": 3, "distance": 3, "yard_line": 41, "special_state": None},
            {"down": 3, "distance": 3, "yard_line": 41, "special_state": None},
        ]

        vi.stabilize_situation_fields(states)

        self.assertIsNone(states[1]["yard_line"])
        self.assertEqual([41, 41, 41], [states[i]["yard_line"] for i in (0, 2, 3)])

    def test_situation_stabilization_keeps_sustained_new_spot(self) -> None:
        states = [
            {"down": 2, "distance": 9, "yard_line": 36, "special_state": None},
            {"down": 2, "distance": 9, "yard_line": 36, "special_state": None},
            {"down": None, "distance": None, "yard_line": 44, "special_state": None},
            {"down": 3, "distance": 0, "yard_line": 44, "special_state": None},
        ]

        vi.stabilize_situation_fields(states)

        self.assertEqual([36, 36, 44, 44], [state["yard_line"] for state in states])
        self.assertEqual(3, states[3]["down"])

    def test_same_clock_situation_collapses_yard_oscillation(self) -> None:
        summaries = [
            {
                "quarter": 2,
                "game_clock": "5:09",
                "down": 1,
                "distance": 10,
                "yard_line": 21,
                "special_state": None,
                "start_timestamp_seconds": 1232.0,
                "end_timestamp_seconds": 1235.0,
                "_prev_off": None,
                "_prev_def": None,
            },
            {
                "quarter": 2,
                "game_clock": "5:09",
                "down": 1,
                "distance": 10,
                "yard_line": 1,
                "special_state": None,
                "start_timestamp_seconds": 1235.0,
                "end_timestamp_seconds": 1245.0,
                "_prev_off": "RPO ZONE ALERT OMAHA",
                "_prev_def": "MIKE BLITZ 0",
            },
        ]

        vi.collapse_same_clock_summaries(summaries)

        self.assertEqual(1, len(summaries))
        self.assertEqual(1, summaries[0]["yard_line"])
        self.assertEqual(1232.0, summaries[0]["start_timestamp_seconds"])
        self.assertEqual(1245.0, summaries[0]["end_timestamp_seconds"])
        self.assertEqual("RPO ZONE ALERT OMAHA", summaries[0]["_prev_off"])

    def test_same_clock_collapses_different_down_reads(self) -> None:
        summaries = [
            {
                "quarter": 2,
                "game_clock": "1:42",
                "down": 1,
                "distance": 10,
                "special_state": None,
                "start_timestamp_seconds": 1500.0,
                "end_timestamp_seconds": 1501.0,
            },
            {
                "quarter": 2,
                "game_clock": "1:42",
                "down": 2,
                "distance": 10,
                "special_state": None,
                "start_timestamp_seconds": 1501.0,
                "end_timestamp_seconds": 1515.0,
            },
        ]

        vi.collapse_same_clock_summaries(summaries)

        self.assertEqual(1, len(summaries))
        self.assertEqual(2, summaries[0]["down"])

    def test_clock_backfill_duplicates_are_collapsed(self) -> None:
        summaries = [
            {
                "quarter": 2,
                "game_clock": "2:57",
                "down": 3,
                "distance": 3,
                "special_state": None,
                "start_timestamp_seconds": 1395.0,
                "end_timestamp_seconds": 1400.0,
            },
            {
                "quarter": 2,
                "game_clock": None,
                "down": 4,
                "distance": 3,
                "special_state": None,
                "start_timestamp_seconds": 1400.0,
                "end_timestamp_seconds": 1415.0,
            },
        ]

        vi.propagate_game_clocks(summaries)
        vi.collapse_same_clock_summaries(summaries)

        self.assertEqual(1, len(summaries))
        self.assertEqual("2:57", summaries[0]["game_clock"])

    def test_clock_increase_is_repaired_before_duplicate_collapse(self) -> None:
        summaries = [
            {"quarter": 4, "game_clock": "0:34"},
            {"quarter": 4, "game_clock": "0:37"},
            {"quarter": 4, "game_clock": "0:34"},
        ]

        vi.repair_quarter_and_clock_progression(summaries)

        self.assertEqual(["0:34", "0:34", "0:34"], [s["game_clock"] for s in summaries])

    def test_clock_restart_advances_delayed_quarter(self) -> None:
        summaries = [
            {"quarter": 1, "game_clock": "0:00"},
            {"quarter": 1, "game_clock": "7:25"},
            {"quarter": 2, "game_clock": "6:53"},
        ]

        vi.repair_quarter_and_clock_progression(summaries)

        self.assertEqual([1, 2, 2], [s["quarter"] for s in summaries])
        self.assertEqual("7:25", summaries[1]["game_clock"])

    def test_low_clock_does_not_prematurely_advance_quarter(self) -> None:
        summaries = [
            {"quarter": 3, "game_clock": "0:07"},
            {"quarter": 4, "game_clock": "0:01"},
            {"quarter": 4, "game_clock": "7:35"},
        ]

        vi.repair_quarter_and_clock_progression(summaries)

        self.assertEqual([3, 3, 4], [s["quarter"] for s in summaries])

    def test_unsupported_same_down_yard_change_is_not_a_play(self) -> None:
        previous = {
            "down": 3,
            "yard_line": 41,
            "special_state": None,
            "away_score": 14,
            "home_score": 14,
            "_banner_shown": False,
        }
        current = {
            "down": 3,
            "yard_line": 47,
            "special_state": None,
            "away_score": 14,
            "home_score": 14,
            "_banner_shown": False,
        }

        self.assertFalse(vi.transition_has_play_evidence(previous, current))

    def test_legal_down_progression_is_play_evidence(self) -> None:
        previous = {"down": 2, "special_state": None}
        current = {"down": 3, "special_state": None, "_banner_shown": False}

        self.assertTrue(vi.transition_has_play_evidence(previous, current))
        self.assertIn(
            "legal_down_progression",
            vi.transition_evidence_reasons(previous, current),
        )

    def test_verified_banner_is_play_evidence_when_situation_ocr_is_incomplete(
        self,
    ) -> None:
        previous = {"down": 2, "special_state": None}
        current = {
            "down": None,
            "special_state": None,
            "_banner_shown": True,
            "_prev_off": "HB STRETCH",
            "_prev_def": None,
        }
        catalog = {"offense": ["HB Stretch"], "defense": []}

        self.assertTrue(vi.transition_has_play_evidence(previous, current, catalog))

    def test_verified_banner_does_not_override_impossible_same_down_transition(
        self,
    ) -> None:
        previous = {"down": 2, "special_state": None}
        current = {
            "down": 2,
            "special_state": None,
            "_banner_shown": True,
            "_prev_off": "MTN SPINNER HB INSIDE",
            "_prev_def": "COVER 3 CLOUD",
        }
        catalog = {
            "offense": ["Mtn Spinner Hb Inside"],
            "defense": ["Cover 3 Cloud"],
        }

        self.assertFalse(vi.transition_has_play_evidence(previous, current, catalog))

    def test_unsupported_summary_is_merged_instead_of_emitted(self) -> None:
        summaries = [
            {
                "down": 3,
                "special_state": None,
                "away_score": 14,
                "home_score": 14,
                "_banner_shown": False,
                "start_timestamp_seconds": 1400.0,
                "end_timestamp_seconds": 1405.0,
            },
            {
                "down": 3,
                "special_state": None,
                "away_score": 14,
                "home_score": 14,
                "_banner_shown": False,
                "start_timestamp_seconds": 1405.0,
                "end_timestamp_seconds": 1410.0,
            },
        ]

        vi.remove_unsupported_situation_transitions(summaries)

        self.assertEqual(1, len(summaries))
        self.assertEqual(1410.0, summaries[0]["end_timestamp_seconds"])

    def test_context_parsers_normalize_observed_ocr(self) -> None:
        self.assertEqual("1 RB, 1 TE, 3 WR", vi.parse_personnel_text("} 1RBJ1TE|3 WR"))
        self.assertEqual(
            "Gun - Normal Y Off Close",
            vi.canonicalize_formation("~] GUN - HORMAL Y OFF CLOSE"),
        )

    def test_field_side_triangle_direction(self) -> None:
        own = np.full((30, 20, 3), 255, dtype=np.uint8)
        opponent = own.copy()
        for row, width in enumerate((12, 10, 8, 6, 4, 2), start=8):
            own[row, :width] = 0
        for row, width in enumerate((2, 4, 6, 8, 10, 12), start=8):
            opponent[row, :width] = 0
        region = vi.Region(0, 0, 1, 1)

        self.assertEqual("own", vi.parse_field_side(own, region))
        self.assertEqual("opponent", vi.parse_field_side(opponent, region))

    def test_possession_dot_selects_team_without_matchup_metadata(self) -> None:
        frame = np.full((100, 100, 3), 220, dtype=np.uint8)
        frame[20:40, 10:20] = 0
        regions = {
            "possession_away": vi.Region(0.1, 0.2, 0.1, 0.2),
            "possession_home": vi.Region(0.7, 0.2, 0.1, 0.2),
        }

        self.assertEqual(
            "AAA",
            vi.parse_possession_team(frame, regions, "AAA", "BBB"),
        )

    def test_possession_flips_to_other_scoreboard_team(self) -> None:
        summaries = [
            {
                "down": 4,
                "special_state": None,
                "away_team": "AAA",
                "home_team": "BBB",
                "possession_team": "AAA",
            },
            {
                "down": 1,
                "special_state": None,
                "away_team": "AAA",
                "home_team": "BBB",
                "possession_team": None,
            },
        ]

        vi.resolve_possession_teams(summaries)

        self.assertEqual("BBB", summaries[1]["possession_team"])

    def test_invalid_possession_ocr_does_not_replace_scoreboard_team(self) -> None:
        summaries = [
            {
                "down": 1,
                "special_state": None,
                "away_team": "NYJ",
                "home_team": "TEN",
                "possession_team": "NYJ",
            },
            {
                "down": 2,
                "special_state": None,
                "away_team": "NYJ",
                "home_team": "TEN",
                "possession_team": "NVJ",
            },
        ]

        vi.resolve_possession_teams(summaries)

        self.assertEqual("NYJ", summaries[1]["possession_team"])

    def test_stream_side_flips_after_possession_change(self) -> None:
        summaries = [
            {"down": 4, "special_state": None, "_stream_side": "defense"},
            {"down": 1, "special_state": None, "_stream_side": "defense"},
            {"down": None, "special_state": "kickoff", "_stream_side": "offense"},
            {"down": 1, "special_state": None, "_stream_side": "offense"},
        ]

        vi.apply_stream_transitions(summaries)

        self.assertEqual("offense", summaries[1]["_stream_side"])
        self.assertEqual("defense", summaries[3]["_stream_side"])

    def test_stream_side_resolves_through_kickoff_without_menu_evidence(self) -> None:
        summaries = [
            {"down": None, "special_state": "extra_point", "_stream_side": "defense"},
            {"down": None, "special_state": "kickoff", "_stream_side": None},
            {"down": 1, "special_state": None, "_stream_side": None},
            {"down": 1, "special_state": None, "_stream_side": "offense"},
        ]

        vi.resolve_stream_sides(summaries)

        self.assertEqual("defense", summaries[1]["_stream_side"])
        self.assertEqual("offense", summaries[2]["_stream_side"])
        self.assertEqual("offense", summaries[3]["_stream_side"])

    def test_extra_point_and_turnover_on_downs_results(self) -> None:
        extra_point = {"play_type": "extra_point", "possession_change": False}
        vi.compute_play_result(
            extra_point,
            {
                "away_score": 0,
                "home_score": 6,
                "down": None,
                "distance": None,
                "yard_line": 15,
            },
            {
                "away_score": 0,
                "home_score": 7,
                "down": None,
                "distance": None,
                "yard_line": 35,
            },
        )
        self.assertEqual("extra point good", extra_point["result"])

        turnover = {"play_type": "pass", "possession_change": True}
        vi.compute_play_result(
            turnover,
            {
                "away_score": 0,
                "home_score": 0,
                "down": 4,
                "distance": 2,
                "yard_line": 43,
            },
            {
                "away_score": 0,
                "home_score": 0,
                "down": 1,
                "distance": 10,
                "yard_line": 43,
            },
        )
        self.assertEqual("turnover on downs", turnover["result"])
        self.assertEqual(0, turnover["yards_gained"])

    def test_following_scoreboard_repairs_hidden_touchdown_and_pat_scores(self) -> None:
        summaries = [
            {
                "special_state": None,
                "possession_team": "TEN",
                "away_team": "NYJ",
                "home_team": "TEN",
                "away_score": 0,
                "home_score": 0,
            },
            {
                "special_state": "extra_point",
                "possession_team": "TEN",
                "away_team": "NYJ",
                "home_team": "TEN",
                "away_score": 0,
                "home_score": 0,
            },
            {
                "special_state": "kickoff",
                "possession_team": "TEN",
                "away_team": "NYJ",
                "home_team": "TEN",
                "away_score": 0,
                "home_score": 0,
            },
            {
                "special_state": None,
                "possession_team": "NYJ",
                "away_team": "NYJ",
                "home_team": "TEN",
                "away_score": 0,
                "home_score": 7,
            },
        ]

        vi.repair_special_team_scores(summaries)

        self.assertEqual(6, summaries[1]["home_score"])
        self.assertEqual(7, summaries[2]["home_score"])

    def test_hidden_failed_pat_keeps_six_point_score_growth(self) -> None:
        summaries = [
            {
                "special_state": None,
                "possession_team": "TEN",
                "away_team": "NYJ",
                "home_team": "TEN",
                "away_score": 21,
                "home_score": 17,
            },
            {
                "special_state": "extra_point",
                "possession_team": "TEN",
                "away_team": "NYJ",
                "home_team": "TEN",
                "away_score": 21,
                "home_score": 17,
            },
            {
                "special_state": "kickoff",
                "possession_team": "TEN",
                "away_team": "NYJ",
                "home_team": "TEN",
                "away_score": 21,
                "home_score": 17,
            },
            {
                "special_state": None,
                "possession_team": "NYJ",
                "away_team": "NYJ",
                "home_team": "TEN",
                "away_score": 21,
                "home_score": 23,
            },
        ]

        vi.repair_special_team_scores(summaries)

        self.assertEqual(23, summaries[1]["home_score"])
        self.assertEqual(23, summaries[2]["home_score"])

    def test_field_goal_score_does_not_report_spot_difference_as_yardage(self) -> None:
        play = {"play_type": None, "possession_change": False}

        vi.compute_play_result(
            play,
            {
                "away_score": 0,
                "home_score": 14,
                "down": 4,
                "distance": 3,
                "yard_line": 12,
            },
            {
                "away_score": 0,
                "home_score": 17,
                "down": None,
                "distance": None,
                "yard_line": 35,
            },
        )

        self.assertEqual("field goal good", play["result"])
        self.assertIsNone(play["yards_gained"])

    def test_play_quality_records_confidence_and_provenance(self) -> None:
        play = {
            "quarter": 1,
            "game_clock": "5:00",
            "away_team": "AAA",
            "home_team": "BBB",
            "away_score": 0,
            "home_score": 0,
            "possession_team": "AAA",
            "down": 1,
            "distance": 10,
            "yard_line": 25,
            "play_type": "run",
            "previous_play_banner_shown": True,
            "offensive_play_call_validation": "verified",
            "defensive_play_call_validation": "verified_fuzzy",
        }

        vi.add_play_quality(
            play,
            {"down": 1, "away_score": 0, "home_score": 0},
            {"down": 2, "away_score": 0, "home_score": 0},
        )

        self.assertGreater(play["overall_confidence"], 0.8)
        self.assertEqual(
            "previous_play_banner+acemadden_catalog", play["call_provenance"]
        )

    def test_extra_point_defensive_ocr_is_not_applicable(self) -> None:
        previous = {
            "start_timestamp_seconds": 1.0,
            "quarter": 1,
            "game_clock": "5:00",
            "away_team": "AAA",
            "away_score": 6,
            "home_team": "BBB",
            "home_score": 0,
            "down": None,
            "distance": None,
            "down_distance": None,
            "yard_line": 15,
            "_stream_side": "offense",
            "possession_team": "AAA",
            "field_side": "opponent",
            "_personnel": None,
            "_formation": None,
            "_banner_shown": False,
            "_prev_off": None,
            "_prev_def": None,
            "special_state": "extra_point",
        }
        current = {
            "start_timestamp_seconds": 5.0,
            "down": None,
            "distance": None,
            "down_distance": None,
            "yard_line": 35,
            "special_state": "kickoff",
            "possession_team": "AAA",
            "_banner_shown": True,
            "_prev_off": None,
            "_prev_def": "OCR NOISE",
            "away_score": 7,
            "home_score": 0,
        }

        play = vi.pair_to_play(previous, current)

        self.assertEqual("not_applicable", play["defensive_play_call_validation"])
        self.assertIsNone(play["defensive_play_call"])

    def test_previous_play_panel_visibility_uses_dark_panel_signature(self) -> None:
        dark_panel = np.zeros((100, 100, 3), dtype=np.uint8)
        dark_panel[:, ::4] = 255
        bright_frame = np.full((100, 100, 3), 180, dtype=np.uint8)
        region = vi.Region(0, 0, 1, 1)

        self.assertTrue(vi.is_previous_play_panel_visible(dark_panel, region))
        self.assertFalse(vi.is_previous_play_panel_visible(bright_frame, region))

    def test_previous_play_heading_requires_both_heading_words(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        region = vi.Region(0, 0, 1, 1)
        with patch.object(
            vi,
            "ocr_text_from_crop",
            side_effect=("PREVIOUS PLAY", "PREVIOUS"),
        ):
            self.assertEqual(
                "PREVIOUS PLAY", vi.read_previous_play_heading(frame, region)
            )
            self.assertIsNone(vi.read_previous_play_heading(frame, region))

    def test_late_down_read_enriches_new_yard_spot(self) -> None:
        states = [
            {
                "quarter": 1,
                "game_clock": "7:51",
                "down": 2,
                "distance": 9,
                "yard_line": 36,
                "away_team": "NYJ",
                "home_team": "TEN",
                "away_score": 0,
                "home_score": 0,
                "prev_off_call": "HB STRETCH",
                "prev_def_call": "COVER 3 SKY",
                "previous_play_banner_shown": True,
            },
            {
                "quarter": 1,
                "game_clock": "7:28",
                "down": None,
                "distance": None,
                "yard_line": 44,
                "away_team": "NYJ",
                "home_team": "TEN",
                "away_score": 0,
                "home_score": 0,
                "prev_off_call": "CHEAT FLAT X POST",
                "prev_def_call": "ROLL 23 CLOUD",
                "previous_play_banner_shown": True,
            },
            {
                "quarter": 1,
                "game_clock": "7:22",
                "down": 3,
                "distance": 0,
                "yard_line": 44,
                "away_team": "NYJ",
                "home_team": "TEN",
                "away_score": 0,
                "home_score": 0,
                "prev_off_call": "CHEAT FLAT X POST",
                "prev_def_call": "ROLL 23 CLOUD",
                "previous_play_banner_shown": True,
            },
        ]
        records = [
            vi.FrameRecord(index, float(index), str(index)) for index in range(3)
        ]

        with patch.object(vi, "extract_frame_state", side_effect=states):
            plays = vi.build_situation_plays(
                records,
                vi.Region(0, 0, 1, 1),
                vi.Region(0, 0, 1, 1),
            )

        self.assertEqual(1, len(plays))
        self.assertEqual(9, plays[0]["yards_gained"])
        self.assertEqual("9-yard completion", plays[0]["result"])
        self.assertEqual("3 & inches", plays[0]["end_down_distance"])


class FrameExtractionTests(unittest.TestCase):
    def test_start_seconds_preserves_absolute_timestamps(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: (
            30.0 if prop == vi.cv2.CAP_PROP_FPS else 0.0
        )
        capture.read.side_effect = [(True, frame), (True, frame), (False, None)]

        with (
            patch.object(vi.cv2, "VideoCapture", return_value=capture),
            patch.object(vi.cv2, "imwrite", return_value=True),
            patch.object(vi.os, "makedirs"),
        ):
            records = vi.extract_frames(
                "video.mp4",
                "output",
                sample_every_n_frames=1,
                max_frames=2,
                start_seconds=120.0,
                max_duration_seconds=10.0,
            )

        self.assertEqual([3600, 3601], [record.frame_index for record in records])
        self.assertEqual(120.0, records[0].timestamp_seconds)
        self.assertAlmostEqual(120.0 + 1 / 30, records[1].timestamp_seconds)


if __name__ == "__main__":
    unittest.main()
