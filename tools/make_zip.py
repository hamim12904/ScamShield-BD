"""Build the exact four-item ScamShield BD submission ZIP.

The ZIP root must contain exactly:
    submission.csv, notebook.ipynb, README.md, .env

Usage (from the repository root):
    python tools/make_zip.py
    python tools/make_zip.py --out NeuraHack_Preli_ScamShield_BD.zip

The script validates ``submission.csv`` with ``tools.validate_submission`` before
and after extraction. It never prints the contents of ``.env``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.validate_submission import validate  # noqa: E402
ZIP_MEMBERS = ("submission.csv", "notebook.ipynb", "README.md", ".env")
DEFAULT_NAME = "NeuraHack_Preli_ScamShield_BD.zip"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_zip(
    out_path: Path,
    submission: Path = ROOT / "submission.csv",
    notebook: Path = ROOT / "notebooks" / "notebook.ipynb",
    readme: Path = ROOT / "README.md",
    env_file: Path = ROOT / ".env",
) -> Path:
    """Create ``out_path`` with exactly the four required root members."""
    sources = {
        "submission.csv": Path(submission),
        "notebook.ipynb": Path(notebook),
        "README.md": Path(readme),
        ".env": Path(env_file),
    }

    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required ZIP item(s): " + ", ".join(missing))

    errors = validate(sources["submission.csv"])
    if errors:
        raise ValueError("submission.csv failed validation:\n  - " + "\n  - ".join(errors))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname in ZIP_MEMBERS:
            zf.write(sources[arcname], arcname=arcname)

    with zipfile.ZipFile(out_path) as zf:
        names = zf.namelist()
        if names != list(ZIP_MEMBERS):
            raise RuntimeError(f"ZIP root must be exactly {list(ZIP_MEMBERS)}, got {names}")

    with tempfile.TemporaryDirectory(prefix="scamshield_zip_") as tmp:
        extract_dir = Path(tmp)
        with zipfile.ZipFile(out_path) as zf:
            zf.extractall(extract_dir)
        extracted = sorted(p.name for p in extract_dir.iterdir())
        if extracted != sorted(ZIP_MEMBERS):
            raise RuntimeError(f"Extracted root mismatch: {extracted}")

        errors = validate(extract_dir / "submission.csv")
        if errors:
            raise ValueError("Extracted submission.csv failed validation:\n  - " + "\n  - ".join(errors))

        for name in ZIP_MEMBERS:
            if _sha256(extract_dir / name) != _sha256(sources[name]):
                raise RuntimeError(f"Extracted {name} differs from the source file")

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the four-item ScamShield BD submission ZIP.")
    parser.add_argument("--out", type=Path, default=ROOT / DEFAULT_NAME, help="Output ZIP path")
    parser.add_argument("--submission", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks" / "notebook.ipynb")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--env", dest="env_file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()

    out = build_zip(args.out, args.submission, args.notebook, args.readme, args.env_file)
    print(f"ZIP: {out}")
    print(f"Size: {out.stat().st_size:,} bytes")
    print(f"SHA-256: {_sha256(out)}")
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            print(f"  {name}  ({zf.getinfo(name).file_size:,} bytes)")
    print("OK: exactly four root items; submission.csv passed strict validation before and after extraction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
