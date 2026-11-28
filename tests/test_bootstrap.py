import importlib.util
import sys
from pathlib import Path
import zipfile
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from bootstrap import unpack_zip, patch_once
from download import fetch


def test_zip_cannot_write_outside_target(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../../outside", "bad")
    with pytest.raises(ValueError, match="escapes"):
        unpack_zip(archive, tmp_path / "target")
    assert not (tmp_path / "outside").exists()


def test_overlay_patch_is_idempotent_and_checks_upstream(tmp_path):
    code = tmp_path / "Main.cs"
    code.write_text("original")
    patch_once(code, "original", "patched")
    patch_once(code, "original", "patched")
    assert code.read_text() == "patched"
    with pytest.raises(ValueError, match="anchor"):
        patch_once(code, "absent", "new")


def test_existing_download_requires_matching_hash(tmp_path):
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum"):
        fetch("https://example.invalid/source.zip", archive, "0" * 64)


def test_visual_rng_patch_does_not_mistake_original_for_replacement(tmp_path):
    source = tmp_path / "Viewmodel.cs"
    source.write_text("class Viewmodel {\n    readonly Random rng = new();\n}")
    original = "\n    readonly Random rng = new();"
    replacement = "\n    Random rng = new();"
    patch_once(source, original, replacement)
    assert "readonly" not in source.read_text()
    patch_once(source, original, replacement)
    assert source.read_text().count("Random rng") == 1
