"""Dataset inspection CLI.

    python -m backend.training.inspect_dataset
    python -m backend.training.inspect_dataset --root /path/to/dataset --limit 200

Writes ``docs/DATASET_INSPECTION.md`` plus a machine-readable JSON sibling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.dataset.inspect import inspect_dataset, render_markdown

DEFAULT_MARKDOWN = Path("docs/DATASET_INSPECTION.md")
DEFAULT_JSON = Path("backend/artifacts/dataset/inspection_report.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the oral-lesion dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root. Defaults to CARESCAN_DATASET_ROOT or autodiscovery.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Probe only the first N images (fast smoke run). Omit for a full pass.",
    )
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--json", dest="json_path", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--no-write", action="store_true", help="Print the report without writing files."
    )
    args = parser.parse_args(argv)

    def say(message: str) -> None:
        print(message, flush=True)

    report, _index, probes = inspect_dataset(
        args.root, probe_limit=args.limit, progress=say
    )
    markdown = render_markdown(report)

    if args.no_write:
        print(markdown)
        return 0

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown, encoding="utf-8")
    report.save_json(args.json_path)

    say("")
    say(f"Probed {len(probes)} image files.")
    say(f"Markdown report -> {args.markdown}")
    say(f"JSON report     -> {args.json_path}")
    if report.n_unreadable:
        say(f"WARNING: {report.n_unreadable} unreadable image file(s).")
    if report.n_duplicate_images:
        say(f"WARNING: {report.n_duplicate_images} pixel-identical duplicate image(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
