"""Regression coverage for path-independent utility helpers."""

import hashlib

from model_quality.utils import load_yaml, sha256_file, write_text


def test_write_text_creates_parent_and_preserves_unicode(tmp_path):
    path = tmp_path / "nested" / "report.txt"
    write_text(path, "test result: success\n")
    assert path.read_text(encoding="utf-8") == "test result: success\n"


def test_load_yaml_reads_explicit_utf8_path(tmp_path):
    source = tmp_path / "config.yaml"
    source.write_text("enabled: true\nalpha: 0.25\n", encoding="utf-8")
    assert load_yaml(source) == {"enabled": True, "alpha": 0.25}


def test_sha256_file_uses_content_hash_and_missing_returns_none(tmp_path):
    source = tmp_path / "bytes.txt"
    source.write_bytes(b"abc")
    assert sha256_file(source) == hashlib.sha256(b"abc").hexdigest()
    assert sha256_file(tmp_path / "missing") is None
