"""MP3 parameter tuning / MP4 RTL editing: python mp3/rtl_run.py --help."""
import argparse
from pathlib import Path

from rtl_backend.config import ROOT, read, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "run", "synth"):
        p = sub.add_parser(command)
        p.add_argument("--design", type=Path, default=ROOT / "configs/rtl_baseline.json")
        p.add_argument("--rtl-root", type=Path, default=ROOT / "rtl")
        p.add_argument("--out", type=Path, required=True)
        if command != "synth":
            p.add_argument("--waves", action="store_true")
        if command == "run":
            p.add_argument("--model", type=Path, default=ROOT / "configs/model_tiny.json")
            p.add_argument("--workload", type=Path, default=ROOT / "configs/workload_smoke.json")
            p.add_argument("--checkpoint", type=Path)
    p = sub.add_parser("compare")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("sweep", help="Evaluate fixed-budget parameter candidates on prefill and decode")
    p.add_argument("--baseline", type=Path, default=ROOT / "configs/rtl_baseline.json")
    p.add_argument("--candidates", type=Path, default=ROOT / "configs/rtl_candidates.json")
    p.add_argument("--model", type=Path, default=ROOT / "configs/model_tiny.json")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sweep":
        from rtl_backend.sweep import execute
        try:
            execute(args)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        return
    if args.command == "compare":
        from rtl_backend.reporting import compare
        compare(args.runs, args.out)
        return
    config = validate(read(args.design))
    if args.command == "synth":
        from rtl_backend.synthesis import synthesize
        print(synthesize(config, args.rtl_root, args.out))
        return
    from rtl_backend.runner import run
    request = dict(mode=args.command, config=config, out=str(args.out.resolve()))
    if args.command == "run":
        request.update(model=read(args.model), workload=read(args.workload),
                       checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None)
    print(run(request, args.rtl_root, args.waves))


if __name__ == "__main__":
    main()
