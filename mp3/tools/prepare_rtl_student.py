"""Create the MP4 editable RTL copy without overwriting student work."""
from pathlib import Path
import argparse
import shutil

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "mp4/rtl_student")
    args = parser.parse_args()
    if args.out.exists():
        baseline = {p.name: p.read_bytes() for p in (ROOT / "mp3/rtl").glob("*.sv")}
        present = {p.name: p.read_bytes() for p in args.out.glob("*.sv")}
        if baseline == present:
            print(f"Unchanged baseline already exists: {args.out}")
            return
        parser.error("Output exists and differs from the baseline; preserving your RTL. Choose another --out.")
    shutil.copytree(ROOT / "mp3/rtl", args.out)
    print(args.out)

if __name__ == "__main__":
    main()
