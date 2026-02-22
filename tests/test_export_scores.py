"""
Tests for 07_export_scores.py -- the per-entity similarity score exporter.

All tests use REAL data (azure_openai.npy, wob_entries.json, etc.).
No mocks. The export script is run once via subprocess and the output
scores.json is loaded and validated by every test class.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# -- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "embeddings_cache"
SCORES_PATH = DATA_DIR / "scores.json"
SCRIPT_PATH = PROJECT_ROOT / "notebooks" / "07_export_scores.py"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python3"

pytestmark = pytest.mark.skipif(
    not (CACHE_DIR / "azure_openai.npy").exists(),
    reason="Real embedding data not available (191 MB .npy file)",
)


# -- Fixture: run the export script once, load scores.json -------------------

@pytest.fixture(scope="module")
def run_export():
    """
    Run 07_export_scores.py via subprocess with default args, then load
    the resulting scores.json.  Runs once per test module.
    """
    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    result = subprocess.run(
        [python, str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT / "notebooks"),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"07_export_scores.py failed (rc={result.returncode}).\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )
    assert SCORES_PATH.exists(), "scores.json was not created"

    with open(SCORES_PATH) as f:
        scores = json.load(f)
    return scores


# ============================================================================
# 1. TestScoresJsonSchema
# ============================================================================

class TestScoresJsonSchema:
    """Verify top-level structure and required fields per entity."""

    def test_top_level_keys(self, run_export):
        assert "meta" in run_export
        assert "entities" in run_export

    def test_meta_fields(self, run_export):
        meta = run_export["meta"]
        assert "model" in meta
        assert "floor" in meta
        assert "total_entries" in meta
        assert "proto_thresholds" in meta

    def test_meta_proto_thresholds(self, run_export):
        pt = run_export["meta"]["proto_thresholds"]
        assert "two" in pt
        assert "three" in pt
        assert pt["two"] < pt["three"]

    def test_meta_total_entries_reasonable(self, run_export):
        total = run_export["meta"]["total_entries"]
        assert total > 10000, f"Expected >10k entries, got {total}"

    def test_entities_is_dict(self, run_export):
        assert isinstance(run_export["entities"], dict)

    def test_entities_not_empty(self, run_export):
        assert len(run_export["entities"]) > 50, (
            f"Expected >50 entities, got {len(run_export['entities'])}"
        )

    def test_entity_required_fields(self, run_export):
        for name, entity in run_export["entities"].items():
            assert "specificity" in entity, f"{name} missing specificity"
            assert "entries_above_floor" in entity, f"{name} missing entries_above_floor"
            assert "calibration" in entity, f"{name} missing calibration"
            assert "prototypes" in entity, f"{name} missing prototypes"
            assert "scores" in entity, f"{name} missing scores"

    def test_calibration_has_percentiles(self, run_export):
        for name, entity in run_export["entities"].items():
            cal = entity["calibration"]
            assert "mean" in cal, f"{name} calibration missing mean"
            assert "std" in cal, f"{name} calibration missing std"
            for pct in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
                key = f"p{pct}"
                assert key in cal, f"{name} calibration missing {key}"

    def test_score_arrays_match_prototype_count(self, run_export):
        """Each entry's score array length must equal the prototype count."""
        for name, entity in run_export["entities"].items():
            n_proto = entity["prototypes"]
            for eid, score_arr in entity["scores"].items():
                assert len(score_arr) == n_proto, (
                    f"{name} entry {eid}: expected {n_proto} scores, "
                    f"got {len(score_arr)}"
                )


# ============================================================================
# 2. TestCalibrationValues
# ============================================================================

class TestCalibrationValues:
    """Verify calibration stats are sensible for a well-known entity."""

    def test_kaladin_percentiles_ordered(self, run_export):
        cal = run_export["entities"]["kaladin"]["calibration"]
        assert cal["p10"] <= cal["p25"] <= cal["p50"], (
            f"Percentiles not ordered: p10={cal['p10']}, p25={cal['p25']}, p50={cal['p50']}"
        )

    def test_kaladin_mean_in_range(self, run_export):
        cal = run_export["entities"]["kaladin"]["calibration"]
        assert 0 < cal["mean"] < 1, f"Mean out of range: {cal['mean']}"

    def test_kaladin_std_positive(self, run_export):
        cal = run_export["entities"]["kaladin"]["calibration"]
        assert cal["std"] > 0, f"Std should be > 0, got {cal['std']}"

    def test_kaladin_p50_above_floor(self, run_export):
        """The median score for explicitly-tagged entries should be well above floor."""
        cal = run_export["entities"]["kaladin"]["calibration"]
        floor = run_export["meta"]["floor"]
        assert cal["p50"] > floor, (
            f"Median calibration score ({cal['p50']}) should exceed floor ({floor})"
        )


# ============================================================================
# 3. TestSpecificityScores
# ============================================================================

class TestSpecificityScores:
    """Verify specificity scores: hubs low, niche entities high."""

    def test_cosmere_low_specificity(self, run_export):
        """Cosmere is a hub topic -- many entries are about it, so specificity is low."""
        spec = run_export["entities"]["cosmere"]["specificity"]
        assert spec < 2.0, f"Cosmere specificity should be < 2.0, got {spec}"

    def test_niche_entities_high_specificity(self, run_export):
        """At least some entities should have notably higher specificity than hubs."""
        # With floor=0.60, niche entities should have much higher specificity
        # than hub topics like cosmere (which is near 0). The max observed
        # specificity at this floor is ~7.91, with 92% of entities above 2.0.
        high_spec = [
            name for name, ent in run_export["entities"].items()
            if ent["specificity"] > 2.0
        ]
        assert len(high_spec) > 0, "No entities found with specificity > 2.0"

    def test_all_specificity_positive(self, run_export):
        for name, entity in run_export["entities"].items():
            assert entity["specificity"] > 0, (
                f"{name} has non-positive specificity: {entity['specificity']}"
            )


# ============================================================================
# 4. TestMultiPrototype
# ============================================================================

class TestMultiPrototype:
    """Verify multi-prototype logic: small entities get 1, large entities may get more."""

    def test_small_entities_single_proto(self, run_export):
        """Entities with 3-4 explicitly-tagged entries should have 1 prototype."""
        single_proto_found = False
        for name, entity in run_export["entities"].items():
            if entity["prototypes"] == 1:
                single_proto_found = True
                break
        assert single_proto_found, "No entities with 1 prototype found"

    def test_at_least_some_multi_proto(self, run_export):
        """At least some entities should have multiple prototypes."""
        multi_proto = [
            name for name, ent in run_export["entities"].items()
            if ent["prototypes"] > 1
        ]
        assert len(multi_proto) > 0, "No entities with multiple prototypes found"

    def test_score_array_length_matches_prototypes(self, run_export):
        """Spot-check: for multi-proto entities, score arrays should be > 1 element."""
        for name, entity in run_export["entities"].items():
            if entity["prototypes"] > 1:
                # Check first score entry
                first_eid = next(iter(entity["scores"]))
                assert len(entity["scores"][first_eid]) == entity["prototypes"]
                break


# ============================================================================
# 5. TestScoreValues
# ============================================================================

class TestScoreValues:
    """Verify score values are within expected ranges."""

    def test_scores_in_valid_range(self, run_export):
        """All scores should be in [-1, 1] (cosine similarity range)."""
        for name, entity in run_export["entities"].items():
            for eid, score_arr in entity["scores"].items():
                for s in score_arr:
                    assert -1.0 <= s <= 1.0, (
                        f"{name} entry {eid} has score {s} outside [-1, 1]"
                    )

    def test_scores_above_floor(self, run_export):
        """All stored scores should have max-across-prototypes > floor."""
        floor = run_export["meta"]["floor"]
        for name, entity in run_export["entities"].items():
            for eid, score_arr in entity["scores"].items():
                max_score = max(score_arr)
                assert max_score > floor - 1e-6, (
                    f"{name} entry {eid}: max score {max_score} not > floor {floor}"
                )

    def test_kaladin_explicit_entries_score_high(self, run_export):
        """
        Kaladin's explicitly-tagged entries should have high similarity.
        With floor=0.60, only entries scoring > 0.60 are stored. Most of
        kaladin's explicitly-tagged entries should be present and score high.
        """
        # Load explicit entry IDs for kaladin
        wob_path = PROJECT_ROOT.parent / "words-of-brandon" / "wob_entries.json"
        with open(wob_path) as f:
            raw_entries = json.load(f)

        kaladin_eids = set()
        for e in raw_entries:
            if "kaladin" in e["tags"]:
                kaladin_eids.add(str(e["id"]))

        kal_scores = run_export["entities"]["kaladin"]["scores"]
        matched = 0
        above_threshold = 0
        for eid in kaladin_eids:
            if eid in kal_scores:
                matched += 1
                if max(kal_scores[eid]) > 0.65:
                    above_threshold += 1

        assert matched > 10, f"Expected >10 matched kaladin entries, got {matched}"
        ratio = above_threshold / matched
        assert ratio > 0.8, (
            f"Only {ratio:.0%} of kaladin explicit entries scored > 0.65 "
            f"(expected > 80%)"
        )


# ============================================================================
# 6. TestOutputFileSize
# ============================================================================

class TestOutputFileSize:
    """Verify scores.json is within size budget."""

    def test_scores_json_under_10mb(self, run_export):
        size_bytes = SCORES_PATH.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        assert size_mb < 10, f"scores.json is {size_mb:.1f} MB, expected < 10 MB"

    def test_scores_json_not_trivially_small(self, run_export):
        size_bytes = SCORES_PATH.stat().st_size
        assert size_bytes > 1000, f"scores.json is suspiciously small: {size_bytes} bytes"
