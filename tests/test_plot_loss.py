import csv
import json
from pathlib import Path
import re

import pytest

from scripts.plot_loss import METRICS, main, read_history, render_report


def history(path, steps=(1000, 2000)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "iteration", "end_step", "finished_at", *("mean_" + key for key in METRICS)])
        writer.writeheader()
        for step in steps:
            writer.writerow({"iteration": 7, "end_step": step,
                             "finished_at": "2026-10-26T12:18:07+00:00",
                             **{"mean_" + key: -3.3 if key == "score" else 0.5
                                for key in METRICS}})
    return path


def test_history_keeps_repeated_iterations_and_negative_score_in_step_order(tmp_path):
    source = history(tmp_path / "training-loss.csv", steps=(2000, 1000, 3000))
    points, warnings = read_history(source)
    assert warnings == []
    assert [point["step"] for point in points] == [1000, 2000, 3000]
    assert [point["iteration"] for point in points] == [7, 7, 7]
    assert all(point["score"] == -3.3 for point in points)


def test_history_reads_completed_rows_during_a_partial_append(tmp_path):
    source = history(tmp_path / "training-loss.csv")
    with source.open("a") as stream:
        stream.write("8,3000,2026-10-26T12:")
    points, warnings = read_history(source)
    assert len(points) == 2
    assert len(warnings) == 1 and "incomplete" in warnings[0]


@pytest.mark.parametrize("replacement", ["nan", "inf", "not-a-number"])
def test_history_reports_invalid_complete_rows(tmp_path, replacement):
    source = history(tmp_path / "training-loss.csv")
    source.write_text(source.read_text().replace(",0.5,", "," + replacement + ",", 1))
    with pytest.raises(ValueError, match="invalid CSV record 1"):
        read_history(source)


def test_history_requires_columns_and_at_least_one_attempt(tmp_path):
    source = tmp_path / "training-loss.csv"
    source.write_text("iteration,end_step\n1,1000\n")
    with pytest.raises(ValueError, match="missing CSV columns"):
        read_history(source)
    history(source, steps=())
    with pytest.raises(ValueError, match="no completed"):
        read_history(source)


def test_report_embeds_real_data_and_escapes_script_delimiters(tmp_path):
    points, _ = read_history(history(tmp_path / "training-loss.csv"))
    report = render_report(points, 50, Path("</script><script>alert(1)</script>"))
    embedded = json.loads(re.search(r"const data = (.+);", report).group(1))
    metadata = json.loads(re.search(r"const metadata = (.+);", report).group(1))
    assert embedded == points
    assert metadata["updated"] == "2026-10-26 20:18:07"
    assert metadata["recent"] == 50 and metadata["attempts"] == 2
    assert "</script><script>alert(1)</script>" not in report
    assert "__LOSS_DATA__" not in report and "__LOSS_METADATA__" not in report
    assert '<script src=' not in report


def test_cli_writes_from_any_directory_and_does_not_change_source(tmp_path, monkeypatch):
    source = history(tmp_path / "run" / "training-loss.csv")
    original = source.read_bytes()
    monkeypatch.chdir(tmp_path)
    assert main(["--root", str(source.parent), "--recent", "1"]) == 0
    report = source.parent / "analysis" / "loss-curve.html"
    assert report.is_file()
    assert source.read_bytes() == original
    assert not list(report.parent.glob(".loss-*"))


def test_cli_rejects_invalid_window_and_source_overwrite(tmp_path):
    source = history(tmp_path / "training-loss.html")
    original = source.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["--csv", str(source), "--output", str(source)])
    assert exc.value.code == 2 and source.read_bytes() == original
    with pytest.raises(SystemExit) as exc:
        main(["--csv", str(source), "--recent", "0"])
    assert exc.value.code == 2
