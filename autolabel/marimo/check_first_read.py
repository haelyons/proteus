"""Focused checks of scientific quantities, reveal ordering and record isolation.

Run with janus/.venv-gpu/bin/python autolabel/marimo/check_first_read.py.
Temporary journals stay under this directory and are removed after each check.
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
import reader


def prediction(unit):
    return {"reader": "AUTOMATED TEST", "prior_exposure": "Known cached reference; not a human reading.",
            "response": "Orientation sensitivity", "evidence": "Janus README R21/R23",
            "rival": "Orientation-invariant response", "confidence": 0.5,
            "transform": "mirror" if unit == "mixed4b:431" else "rotate_180",
            "low": 0.1, "high": 0.6, "category": reader.CATEGORIES[0]}


class FirstReadChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packets = {unit: reader.load_packet(unit) for unit in reader.UNITS}

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=reader.HERE)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_two_units_and_expected_gaps(self):
        dog, contested = (self.packets[u] for u in reader.UNITS)
        self.assertEqual([p["n_card"] for p in (dog, contested)], [7038, 7038])
        self.assertEqual([len(p["profile"]) for p in (dog, contested)], [10, 10])
        self.assertEqual(len(dog["pair_rows"]), 24)
        self.assertFalse(dog["wiring"])
        self.assertFalse(contested["pair_rows"])
        self.assertTrue(contested["wiring"])
        self.assertTrue(contested["images"]["negative"].is_file())

    def test_profile_matches_separately_cached_stratum(self):
        for packet in self.packets.values():
            with (reader.JANUS / f"runs/rangeprofile/{packet['layer']}.csv").open() as stream:
                cached = next(r for r in csv.DictReader(stream) if r["unit"] == packet["unit"])
            for row in packet["profile"]:
                self.assertAlmostEqual(row["retention"], float(cached[f"{row['transform']}_ret_s1"]), places=5)
            result = reader.score_prediction(packet, prediction(packet["unit"]))
            self.assertTrue(result["interval_hit"])
            self.assertTrue(result["category_hit"])

    def test_pair_reference_sign_and_replication(self):
        row = self.packets["mixed4b:431"]["pair_rows"][0]
        self.assertEqual((row["higher on D"], row["lower on D"]), ("right", "left"))
        self.assertAlmostEqual(row["T percentile"], 97.8516, places=4)
        self.assertEqual(row["T outcome"], "held")
        self.assertEqual(row["T_null outcome"], "not held")

    def test_order_lock_idempotence_and_revision(self):
        packet = self.packets["mixed4b:431"]
        session = reader.ReadSession(packet, self.directory)
        revision = {key: "AUTOMATED TEST; not a substantive reading" for key in
                    ("response", "wiring", "family", "failures", "evidence", "missing_evidence", "next_test")}
        revision.update({"class": "cannot_tell", "confidence": 0.5})
        with self.assertRaises(ValueError):
            session.save_reading(revision)
        self.assertFalse(list(self.directory.iterdir()))
        fields = prediction(packet["unit"])
        receipt = session.commit_and_reveal(fields)
        self.assertEqual(len(list(self.directory.iterdir())), 2)
        pred = receipt["prediction"]
        outcome = receipt["outcome"]
        original = Path(pred["path"]).read_bytes()
        self.assertLessEqual(pred["record"]["utc"], outcome["record"]["utc"])
        self.assertEqual(reader.digest(pred["path"]), outcome["record"]["prediction_sha256"])
        self.assertEqual(session.commit_and_reveal(fields), receipt)
        with self.assertRaises(ValueError):
            session.commit_and_reveal({**fields, "response": "Changed after reveal"})
        saved = session.save_reading(revision)
        self.assertEqual(session.save_reading(revision), saved)
        self.assertEqual(len(list(self.directory.iterdir())), 3)
        self.assertEqual(Path(pred["path"]).read_bytes(), original)
        self.assertEqual(json.loads(Path(saved["path"]).read_text())["fields"]["class"], "cannot_tell")

    def test_units_have_separate_records(self):
        paths = []
        for unit, packet in self.packets.items():
            receipt = reader.ReadSession(packet, self.directory).commit_and_reveal(prediction(unit))
            paths.append(receipt["prediction"]["path"])
            self.assertEqual(receipt["prediction"]["record"]["unit"], unit)
        self.assertNotEqual(*paths)
        self.assertEqual(len(list(self.directory.iterdir())), 4)

    def test_bad_or_missing_measurements_never_pass(self):
        fields = prediction("mixed4b:431")
        self.assertIsNone(reader.validate_prediction({**fields, "transform": [fields["transform"]],
                                                     "category": [fields["category"]]}))
        for change in ({"high": -1}, {"low": float("nan")}, {"reader": " "}, {"confidence": 2}):
            self.assertIsNotNone(reader.validate_prediction({**fields, **change}))
        result = reader.score_prediction({"profile": []}, fields)
        self.assertEqual(result, {"status": "unavailable", "interval_hit": None, "category_hit": None})

    def test_write_failure_does_not_reveal(self):
        blocked = self.directory / "not_a_directory"
        blocked.write_text("test")
        session = reader.ReadSession(self.packets["mixed4b:431"], blocked)
        with self.assertRaises(OSError):
            session.commit_and_reveal(prediction("mixed4b:431"))
        self.assertIsNone(session.prediction)
        self.assertIsNone(session.outcome)

    def test_changed_source_rejected_and_writes_confined(self):
        source = self.directory / "source.json"
        source.write_text("before")
        packet = dict(self.packets["mixed4b:431"])
        packet["manifest"] = {str(source.relative_to(reader.ROOT)): {"sha256": reader.digest(source)}}
        session = reader.ReadSession(packet, self.directory / "journal")
        source.write_text("after")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            session.commit_and_reveal(prediction(packet["unit"]))
        self.assertFalse(session.directory.exists())
        with self.assertRaises(ValueError):
            reader.ReadSession(packet, reader.JANUS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
