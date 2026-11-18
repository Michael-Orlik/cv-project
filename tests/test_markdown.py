import sys

from PIL import Image

from main import main, save_markdown


def test_save_markdown_creates_a_nested_utf8_file(tmp_path):
    destination = tmp_path / "nested" / "result.md"

    save_markdown(["First line.", "An en dash: \u2013"], destination)

    assert destination.read_bytes() == "First line.\nAn en dash: \u2013\n".encode("utf-8")


def test_blank_page_replaces_old_text_without_loading_ocr(tmp_path, monkeypatch, capsys):
    source = tmp_path / "blank.png"
    destination = tmp_path / "result.md"
    Image.new("RGB", (200, 300), "white").save(source)
    destination.write_text("Old recognition result.", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "main.py", str(source), str(destination),
        "--segment-page", "--recognize-page", "--lines-output", str(tmp_path / "lines"),
    ])
    # Importing OCR would fail if the empty-page shortcut stopped working.
    monkeypatch.setitem(sys.modules, "recognize", None)

    assert main() == 0
    assert destination.read_bytes() == b""
    assert "No text lines found." in capsys.readouterr().out
