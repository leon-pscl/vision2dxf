#!/usr/bin/env python3
"""pattern_drafter command line interface.

Draft sewing patterns from YAML measurement files using the OpenPattern
library, then export them as structured JSON and a visual PDF.

Examples
--------
    python app.py templates
    python app.py check examples/bodice.yaml
    python app.py build examples/bodice.yaml examples/skirt.yaml
    python app.py build --outdir outputs examples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from pattern_drafter.drafting import OpenPatternMissing
from pattern_drafter.measurements import (
    MeasurementError,
    TEMPLATES,
    load_measurement_file,
    validate_measurement_file,
)
from pattern_drafter.pipeline import iter_yaml_files, process_file

DEFAULT_OUTDIR = "outputs"
DEFAULT_FORMATS = "json,pdf"


def _print_result(result) -> None:
    data = result.data
    bounds = data["bounds"]
    print(
        "  + %-24s template=%-9s style=%-11s panels=%d points=%d"
        % (
            result.config["profile_name"],
            result.config["template"],
            result.config["style"],
            len(data["panels"]),
            sum(len(g["points"]) for g in data["points"].values()),
        )
    )
    print(
        "    bounds: x[%.1f, %.1f] y[%.1f, %.1f] cm"
        % (bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"])
    )
    for path in result.written:
        print("    -> %s" % path)


def cmd_build(args: argparse.Namespace) -> int:
    try:
        files = iter_yaml_files(args.inputs)
    except MeasurementError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    if not files:
        print("error: no YAML files found", file=sys.stderr)
        return 2

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    invalid = set(formats) - {"json", "pdf"}
    if invalid:
        print(
            "error: unknown format(s): %s" % ", ".join(sorted(invalid)),
            file=sys.stderr,
        )
        return 2

    print("Drafting %d file(s) into %s" % (len(files), out_dir))
    ok = 0
    for yaml_path in files:
        print("- %s" % yaml_path)
        try:
            result = process_file(yaml_path, out_dir, formats)
        except MeasurementError as exc:
            print("  ! %s" % exc)
            continue
        except OpenPatternMissing as exc:
            print("error: %s" % exc, file=sys.stderr)
            return 3
        _print_result(result)
        ok += 1

    print("\nDone: %d/%d succeeded." % (ok, len(files)))
    return 0 if ok == len(files) else 1


def cmd_check(args: argparse.Namespace) -> int:
    try:
        files = iter_yaml_files(args.inputs)
    except MeasurementError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    status = 0
    for yaml_path in files:
        try:
            config = load_measurement_file(yaml_path)
            validate_measurement_file(config)
        except MeasurementError as exc:
            print("FAIL %s\n  %s" % (yaml_path, exc))
            status = 1
        else:
            print("OK   %s" % yaml_path)
    return status


def cmd_templates(args: argparse.Namespace) -> int:
    for name, spec in sorted(TEMPLATES.items()):
        styles = sorted({style for (style, _) in spec["requirements"]})
        print("%s — %s" % (name, spec["description"]))
        print(
            "    class=%s default=%s/%s styles=%s"
            % (
                spec["op_class"],
                spec["default_gender"],
                spec["default_style"],
                ", ".join(styles),
            )
        )
        if spec["one_of"]:
            for group in spec["one_of"]:
                print("    needs one of: %s" % ", ".join(group))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Draft sewing patterns from YAML measurements with OpenPattern.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="draft pattern(s) and export JSON/PDF")
    build.add_argument(
        "inputs",
        nargs="+",
        help="YAML measurement file(s) or directories containing them",
    )
    build.add_argument(
        "-o",
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="output directory (default: %s)" % DEFAULT_OUTDIR,
    )
    build.add_argument(
        "--formats",
        default=DEFAULT_FORMATS,
        help="comma separated list of json,pdf (default: %s)" % DEFAULT_FORMATS,
    )
    build.set_defaults(func=cmd_build)

    check = sub.add_parser("check", help="validate YAML file(s) only")
    check.add_argument("inputs", nargs="+", help="YAML file(s) or directories")
    check.set_defaults(func=cmd_check)

    templates = sub.add_parser("templates", help="list supported base templates")
    templates.set_defaults(func=cmd_templates)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
