"""List historical index manifests and their source refs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi_docs_rag.indexing.manifest import IndexManifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/index_manifest.json"),
    )
    args = parser.parse_args()

    history_dir = args.manifest.parent / "history"
    manifests = [
        IndexManifest.load(path)
        for path in sorted(history_dir.glob("*.json"))
    ] if history_dir.exists() else []
    current = IndexManifest.load(args.manifest)

    print(f"live: {current.build_id if current else 'none'}")
    print(f"history_dir: {history_dir}")
    for manifest in manifests:
        if manifest is None:
            continue
        marker = " <- live" if current and manifest.build_id == current.build_id else ""
        print(
            f"{manifest.build_id}  {manifest.source_ref}  "
            f"v{manifest.docs_version}  {manifest.build_collection}{marker}"
        )


if __name__ == "__main__":
    main()
