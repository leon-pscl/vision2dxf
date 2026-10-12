"""Provision the station's model weights, once, deliberately.

The application never downloads anything (brief 5): a station is provisioned
before it is used, and from then on startup only checks that the files on disk
are the ones the configuration names. This script is that provisioning step, run
by hand on a machine that is allowed to reach the network.

    python tools/provision_models.py                 # report what is there
    python tools/provision_models.py --fetch         # download the missing files
    python tools/provision_models.py --write-hashes  # print the config block to paste
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bodymeasure.config import load_app_config  # noqa: E402
from bodymeasure.vision import file_sha256, model_paths  # noqa: E402


def fetch(path: Path) -> None:
    """Download one weight file by name into its configured folder.

    Raises:
        RuntimeError: if ultralytics is not installed, which is the only thing
            here that knows how to find a published weight file.

    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - a provisioning-only path
        raise RuntimeError(
            "ultralytics is not installed, so this script cannot fetch weights. "
            "Install it, or copy the weight files in by hand."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {path.name}…")
    YOLO(path.name)  # downloads into the working directory
    downloaded = Path(path.name)
    if downloaded.exists() and downloaded.resolve() != path.resolve():
        downloaded.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Report, fetch or hash the configured weights. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download any missing weight file")
    parser.add_argument(
        "--write-hashes", action="store_true", help="print the models: block to paste into config"
    )
    arguments = parser.parse_args(argv)

    models = load_app_config().models
    seg_path, pose_path = model_paths()
    expected = {"seg": models.seg_sha256, "pose": models.pose_sha256}
    digests: dict[str, str] = {}
    problems = 0

    for name, path in (("seg", seg_path), ("pose", pose_path)):
        if not path.exists() and arguments.fetch:
            fetch(path)
        if not path.exists():
            print(f"{name}: MISSING {path} — run with --fetch, or copy the file in")
            problems += 1
            continue
        digests[name] = file_sha256(path)
        want = expected[name]
        if want is None:
            print(f"{name}: {path.name} present, sha256 {digests[name]} (no hash configured)")
        elif want == digests[name]:
            print(f"{name}: {path.name} present and matches the configured hash")
        else:
            print(f"{name}: {path.name} sha256 {digests[name]} does NOT match configured {want}")
            problems += 1

    if arguments.write_hashes and digests:
        print("\nPaste into config/app.yaml under models:")
        for name, digest in digests.items():
            print(f"  {name}_sha256: \"{digest}\"")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
