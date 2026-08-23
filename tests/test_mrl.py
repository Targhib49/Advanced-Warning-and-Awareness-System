"""Tests for MRL parsing and the subject-wise split.

These deliberately import no torch: label parsing and the split are where a
silent mistake would poison every downstream number, so they must be testable
on their own.
"""

from pathlib import Path

import pytest

from training.mrl import (
    EyeSample,
    MRLFilenameError,
    describe,
    parse_filename,
    split_by_subject,
)


def sample(subject="s0001", imgno="00001", gender="0", glasses="0",
           state="1", refl="0", light="1", sensor="01"):
    name = f"{subject}_{imgno}_{gender}_{glasses}_{state}_{refl}_{light}_{sensor}.png"
    return parse_filename(Path(subject) / name)


class TestParseFilename:
    def test_reads_every_field(self):
        s = sample(subject="s0007", glasses="1", state="0", refl="2", light="0", sensor="03")
        assert s.subject == "s0007"
        assert s.label == 0            # closed
        assert s.glasses is True
        assert s.lighting_good is False
        assert s.reflections == 2
        assert s.sensor == "03"

    def test_open_eye_is_label_one(self):
        assert sample(state="1").label == 1

    def test_closed_eye_is_label_zero(self):
        assert sample(state="0").label == 0

    def test_strata_are_named(self):
        s = sample(glasses="1", light="0", refl="1", sensor="02")
        assert s.strata == {
            "glasses": "glasses",
            "lighting": "bad_light",
            "reflections": "small",
            "sensor": "ids",
        }

    @pytest.mark.parametrize("code,name", [
        ("01", "realsense"), ("02", "ids"), ("03", "aptina"),
    ])
    def test_sensor_ids_are_named(self, code, name):
        assert sample(sensor=code).sensor_name == name

    def test_unknown_sensor_degrades_readably(self):
        assert sample(sensor="09").sensor_name == "sensor_09"

    @pytest.mark.parametrize("bad", [
        "s0001_00001_0_0_0_0_0.png",        # too few fields
        "0001_00001_0_0_0_0_0_01.png",      # no subject prefix
        "s0001_00001_0_0_x_0_0_01.png",     # non-numeric state
        "random_image.png",
    ])
    def test_rejects_malformed_names(self, bad):
        with pytest.raises(MRLFilenameError):
            parse_filename(Path(bad))

    def test_error_message_is_actionable(self):
        with pytest.raises(MRLFilenameError, match="naming convention"):
            parse_filename(Path("nope.png"))


def make_samples(n_subjects=10, per_subject=20):
    out = []
    for i in range(n_subjects):
        subj = f"s{i:04d}"
        for j in range(per_subject):
            out.append(sample(subject=subj, imgno=f"{j:05d}", state=str(j % 2)))
    return out


class TestSubjectWiseSplit:
    def test_no_subject_appears_in_two_splits(self):
        """The whole point: a driver in test must be unseen in training."""
        tr, va, te = split_by_subject(make_samples(), 0.15, 0.2, seed=1)
        s_tr = {s.subject for s in tr}
        s_va = {s.subject for s in va}
        s_te = {s.subject for s in te}
        assert not (s_tr & s_va)
        assert not (s_tr & s_te)
        assert not (s_va & s_te)

    def test_every_sample_lands_somewhere(self):
        samples = make_samples()
        tr, va, te = split_by_subject(samples, 0.15, 0.2, seed=1)
        assert len(tr) + len(va) + len(te) == len(samples)

    def test_all_splits_are_populated(self):
        tr, va, te = split_by_subject(make_samples(), 0.15, 0.2, seed=1)
        assert tr and va and te

    def test_is_deterministic_for_a_seed(self):
        a = split_by_subject(make_samples(), 0.15, 0.2, seed=42)
        b = split_by_subject(make_samples(), 0.15, 0.2, seed=42)
        assert [s.path for s in a[2]] == [s.path for s in b[2]]

    def test_different_seeds_can_differ(self):
        a = split_by_subject(make_samples(30), 0.15, 0.2, seed=1)
        b = split_by_subject(make_samples(30), 0.15, 0.2, seed=2)
        assert {s.subject for s in a[2]} != {s.subject for s in b[2]}

    def test_refuses_a_split_with_too_few_subjects(self):
        with pytest.raises(ValueError, match="at least 3"):
            split_by_subject(make_samples(n_subjects=2), 0.15, 0.2, seed=1)


class TestDescribe:
    def test_reports_counts(self):
        line = describe(make_samples(4, 10), "train")
        assert "40 images" in line
        assert "4 subjects" in line

    def test_handles_empty(self):
        assert describe([], "test") == "test: empty"
