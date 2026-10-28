#!/usr/bin/env python3
"""Plot completed training attempts as a self-contained, offline HTML report."""

import argparse
import csv
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import webbrowser
from zoneinfo import ZoneInfo


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "runs" / "zero-9x9"
TEMPLATE = Path(__file__).with_name("loss_curve.html")
METRICS = ("total", "policy", "value", "score", "ownership", "moves", "grad_norm")
REQUIRED = ("iteration", "end_step", "finished_at", *("mean_" + k for k in METRICS))


def read_history(path):
    """Read one CSV snapshot, keeping separate attempts for repeated iterations."""
    source = Path(path).read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(source), strict=True)
    missing = set(REQUIRED) - set(reader.fieldnames or ())
    if missing:
        raise ValueError("missing CSV columns: " + ", ".join(sorted(missing)))
    rows = list(reader)
    points = []
    warnings = []
    for index, row in enumerate(rows):
        try:
            if None in row:
                raise ValueError("too many CSV fields")
            iteration, step = int(row["iteration"]), int(row["end_step"])
            if iteration < 0 or step < 1:
                raise ValueError("iteration must be nonnegative and end_step positive")
            finished = datetime.fromisoformat(row["finished_at"])
            if finished.tzinfo is None:
                raise ValueError("finished_at must include a timezone")
            values = {key: float(row["mean_" + key]) for key in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("loss metrics must be finite")
            points.append({"iteration": iteration, "step": step,
                           "finished_at": finished.isoformat(), **values})
        except (TypeError, ValueError) as exc:
            # A running loop can be in the middle of appending its final CSV row.
            if index == len(rows) - 1 and not source.endswith(("\n", "\r")):
                warnings.append(f"ignored incomplete final CSV row: {exc}")
                continue
            raise ValueError(f"invalid CSV record {index + 1}: {exc}") from exc
    if not points:
        raise ValueError("no completed training attempts in CSV")
    # Use steps rather than iteration numbers: reruns can repeat an iteration.
    points.sort(key=lambda point: point["step"])
    return points, warnings


def render_report(points, recent, source):
    template = TEMPLATE.read_text(encoding="utf-8")
    latest = points[-1]
    finished = datetime.fromisoformat(latest["finished_at"]).astimezone(ZoneInfo("Asia/Shanghai"))
    metadata = {"source": str(source), "recent": recent,
                "updated": finished.strftime("%Y-%m-%d %H:%M:%S"),
                "iteration": latest["iteration"], "step": latest["step"],
                "attempts": len(points)}
    # Escape script delimiters even if a source path contains HTML-like text.
    def payload(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).replace("<", "\\u003c")
    return template.replace("__LOSS_DATA__", payload(points)).replace(
        "__LOSS_METADATA__", payload(metadata))


def write_report(output, content):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".loss-", suffix=".html", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="run directory (default: project/runs/zero-9x9)")
    parser.add_argument("--csv", type=Path, help="read a specific training-loss.csv instead")
    parser.add_argument("--output", type=Path,
                        help="output HTML (default: CSV_PARENT/analysis/loss-curve.html)")
    parser.add_argument("--recent", type=int, default=50,
                        help="attempts in the recent-loss panel (default: 50)")
    parser.add_argument("--open", action="store_true", help="open the report in your browser")
    args = parser.parse_args(argv)
    if args.recent < 1:
        parser.error("--recent must be positive")
    source = (args.csv or args.root / "training-loss.csv").expanduser().resolve()
    output = (args.output or source.parent / "analysis" / "loss-curve.html").expanduser().resolve()
    if output == source:
        parser.error("output must not overwrite the source CSV")
    if output.suffix.lower() != ".html":
        parser.error("--output must have an .html extension")
    try:
        points, warnings = read_history(source)
        report = render_report(points, args.recent, source)
        write_report(output, report)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    for warning in warnings:
        print("Warning: " + warning, file=sys.stderr)
    latest = points[-1]
    print(f"Read {len(points)} attempts: step {points[0]['step']:,}–{latest['step']:,}")
    print(f"Latest iteration {latest['iteration']}: total={latest['total']:.6f}, "
          f"policy={latest['policy']:.6f}")
    print(f"Report: {output}")
    if args.open and not webbrowser.open(output.as_uri()):
        print("Warning: could not open browser; open the report path manually.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
