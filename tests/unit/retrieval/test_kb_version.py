"""Tests for the knowledge-base version stamp."""

from __future__ import annotations

from app.retrieval.kb_version import KnowledgeVersion


def test_starts_at_zero(tmp_path):
    assert KnowledgeVersion(persist_dir=str(tmp_path)).current() == 0


def test_bump_increments_and_persists(tmp_path):
    version = KnowledgeVersion(persist_dir=str(tmp_path))

    assert version.bump() == 1
    assert version.bump() == 2

    reloaded = KnowledgeVersion(persist_dir=str(tmp_path))
    assert reloaded.current() == 2


def test_corrupt_file_reads_as_zero(tmp_path):
    (tmp_path / "kb_version.txt").write_text("not-a-number", encoding="utf-8")

    assert KnowledgeVersion(persist_dir=str(tmp_path)).current() == 0
