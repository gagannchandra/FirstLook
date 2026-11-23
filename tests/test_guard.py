"""The S1 test set must be unreachable until choices are frozen."""
import json

import pytest

from src import common


def test_test_rows_sealed_without_freeze(monkeypatch, tmp_path):
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"date_frozen": None, "tiers": {}}))
    monkeypatch.setattr(common, "SEL", sel)
    monkeypatch.setenv("FIRSTLOOK_OPEN_TEST", "1")
    with pytest.raises(common.TestSetSealed):
        common.assert_test_allowed()


def test_test_rows_sealed_without_env(monkeypatch, tmp_path):
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"date_frozen": "2026-11-24", "tiers": {}}))
    monkeypatch.setattr(common, "SEL", sel)
    monkeypatch.delenv("FIRSTLOOK_OPEN_TEST", raising=False)
    with pytest.raises(common.TestSetSealed):
        common.assert_test_allowed()


def test_open_when_frozen_and_flag(monkeypatch, tmp_path):
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"date_frozen": "2026-11-24", "tiers": {}}))
    monkeypatch.setattr(common, "SEL", sel)
    monkeypatch.setenv("FIRSTLOOK_OPEN_TEST", "1")
    common.assert_test_allowed()
