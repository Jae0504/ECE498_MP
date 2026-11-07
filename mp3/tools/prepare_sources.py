"""Prepare patched runtime copies of pinned, publicly available Git submodules."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile

MP3_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MP3_ROOT.parent
DEPENDENCIES = MP3_ROOT / ".deps"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_manifest(root):
    return {str(path.relative_to(root)): file_hash(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def git(*args, cwd=REPO_ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def verify_prepared(expected):
    marker = DEPENDENCIES / "manifest.json"
    if not marker.is_file():
        raise ValueError("Prepared sources are missing. Run: python3 mp3/tools/prepare_sources.py")
    saved = json.loads(marker.read_text())
    if any(saved.get("inputs", {}).get(name) != spec for name, spec in expected.items()):
        raise ValueError("Prepared sources use different revisions/patches. Preserve local edits before preparing fresh sources.")
    for name in expected:
        if file_manifest(DEPENDENCIES / name) != saved.get("files", {}).get(name):
            raise ValueError(f"Prepared baseline {name} has changed. Preserve your edits in mp4/; move mp3/.deps aside and rerun setup.")


def extract_source(source, spec, destination):
    destination.mkdir()
    # Read committed files, not local untracked files. No tar executable is required.
    with tempfile.TemporaryFile() as stream:
        subprocess.run(["git", "-C", str(source), "archive", spec["revision"], "--", *spec["paths"]],
                       stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            for entry in archive:
                relative = PurePosixPath(entry.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unexpected path in upstream source archive")
                target = destination.joinpath(*relative.parts)
                if entry.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif entry.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(entry) as reader, target.open("wb") as writer:
                        shutil.copyfileobj(reader, writer)
                    target.chmod(entry.mode & 0o777)
                else:
                    raise ValueError(f"Unsupported upstream archive entry: {entry.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only verify the existing prepared sources")
    parser.add_argument("--only", nargs="+", help="Prepare only named sources; the RTL path needs miniLLAMA")
    args = parser.parse_args()
    try:
        specs = json.loads((MP3_ROOT / "vendor_versions.json").read_text())
        if args.only:
            if set(args.only) - set(specs):
                raise ValueError(f"Unknown source names: {set(args.only) - set(specs)}")
            specs = {name: specs[name] for name in args.only}
        expected = {name: {"revision": spec["revision"], "patch_sha256": file_hash(MP3_ROOT / spec["patch"])}
                    for name, spec in specs.items()}
        for name, spec in specs.items():
            if spec.get("overlays"):
                expected[name]["overlay_sha256"] = {
                    destination: file_hash(MP3_ROOT / source)
                    for destination, source in spec["overlays"].items()
                }
        if args.check:
            verify_prepared(expected)
            print("Prepared course sources: PASS (unchanged)")
            return 0
        marker = DEPENDENCIES / "manifest.json"
        saved = json.loads(marker.read_text()) if marker.is_file() else {"inputs": {}, "files": {}}
        existing = {name: spec for name, spec in expected.items() if name in saved["inputs"]}
        if existing:
            verify_prepared(existing)
        specs = {name: spec for name, spec in specs.items() if name not in existing}
        if not specs:
            print("Prepared course sources: PASS (unchanged)")
            return 0
        for name in specs:
            if (DEPENDENCIES / name).exists():
                raise ValueError(f"Unrecorded source directory exists: {DEPENDENCIES / name}; preserving its contents")
        missing = [name for name in specs if not (REPO_ROOT / name / ".git").exists()]
        if missing:
            subprocess.run(["git", "-C", str(REPO_ROOT), "submodule", "update", "--init", "--", *missing], check=True)
        for name, spec in specs.items():
            source = REPO_ROOT / name
            if Path(git("rev-parse", "--show-toplevel", cwd=source)).resolve() != source.resolve():
                raise ValueError(f"{name} is not an initialized submodule")
            if git("rev-parse", "HEAD", cwd=source) != spec["revision"]:
                raise ValueError(f"{name} is at a different revision; initialize the pinned course submodules first")
            if subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"]).returncode:
                raise ValueError(f"{name} has local source edits; preserve them separately before preparing the course baseline")
        DEPENDENCIES.mkdir(parents=True, exist_ok=True)
        # Apply patches outside the checkout. Inside a Git subdirectory, Git's
        # path prefix can silently exclude a patch whose paths are model.py, etc.
        with tempfile.TemporaryDirectory(prefix="ece498-mp3-") as temporary:
            prepared = Path(temporary) / "prepared"
            prepared.mkdir()
            for name, spec in specs.items():
                destination = prepared / name
                extract_source(REPO_ROOT / name, spec, destination)
                patch = MP3_ROOT / spec["patch"]
                subprocess.run(["git", "apply", "--no-index", "--check", str(patch)], cwd=destination, check=True)
                subprocess.run(["git", "apply", "--no-index", str(patch)], cwd=destination, check=True)
                for target, source in spec.get("overlays", {}).items():
                    shutil.copy2(MP3_ROOT / source, destination / target)
            for name in specs:
                saved["inputs"][name] = expected[name]
                saved["files"][name] = file_manifest(prepared / name)
                shutil.move(str(prepared / name), str(DEPENDENCIES / name))
            with tempfile.NamedTemporaryFile(mode="w", prefix=".manifest-", dir=DEPENDENCIES,
                                             delete=False) as stream:
                stream.write(json.dumps(saved, indent=2) + "\n")
                staged_manifest = Path(stream.name)
            staged_manifest.replace(marker)
        print(f"Prepared course sources: {DEPENDENCIES}")
        print("The upstream submodules remain unchanged. No sudo or compiler was used.")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
