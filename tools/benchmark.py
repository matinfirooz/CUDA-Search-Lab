#!/usr/bin/env python3
"""Reproducible end-to-end sweeps. No third-party Python dependencies."""
import argparse
import datetime
import json
import pathlib
import platform
import shutil
import subprocess
import sys


def capture(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", default="build/searchlab")
    p.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--sizes", default="16,32,64")
    p.add_argument("--algorithms", default="bfs,dijkstra,astar,weighted-astar,greedy,bellman-ford,delta-stepping")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results/sweep"))
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    exe = pathlib.Path(args.exe).resolve()
    if not exe.is_file():
        p.error("executable does not exist: " + str(exe))
    if args.repeat < 1 or args.warmup < 0:
        p.error("repeat >=1 and warmup >=0 are required")
    sizes = [int(s) for s in args.sizes.split(",")]
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(), "python": sys.version,
        "cpu": platform.processor(), "backend": args.backend,
        "timing": "median complete CLI search-call wall time; per-process warmups",
        "cuda_runtime_measured": args.backend == "cuda",
        "seed": args.seed, "repeat": args.repeat, "warmup": args.warmup,
        "nvidia_smi": capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        if shutil.which("nvidia-smi") else None,
        "commands": [],
    }
    all_rows = []
    for size in sizes:
        for algorithm in algorithms:
            stem = args.output / (str(size) + "_" + algorithm)
            command = [
                str(exe), "--backend", args.backend, "--algo", algorithm,
                "--random-grid", str(size), str(size), "--seed", str(args.seed),
                "--repeat", str(args.repeat), "--warmup", str(args.warmup),
                "--json", str(stem) + ".json", "--csv", str(stem) + ".csv",
            ]
            if args.validate:
                command.append("--validate")
            metadata["commands"].append(command)
            print("Running", size, algorithm, flush=True)
            subprocess.run(command, check=True)
            rows = json.loads(pathlib.Path(str(stem) + ".json").read_text())
            for row in rows:
                row.update(grid_width=size, grid_height=size, seed=args.seed)
            all_rows.extend(rows)
    (args.output / "combined.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
