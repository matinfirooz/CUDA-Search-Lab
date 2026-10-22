#!/usr/bin/env python3
"""Integration checks for exported data and rejected CLI inputs."""
import argparse
import csv
import json
import pathlib
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exe", default="build-cpu/searchlab")
    args = p.parse_args()
    exe = str(pathlib.Path(args.exe).resolve())
    root = pathlib.Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        subprocess.run([exe, "--backend", "cpu", "--algo", "all", "--graph",
                        str(root / "examples/weighted.edgelist"), "--validate",
                        "--json", str(out / "r.json"), "--csv", str(out / "r.csv")], check=True,
                       stdout=subprocess.DEVNULL)
        rows = json.loads((out / "r.json").read_text())
        csv_rows = list(csv.DictReader((out / "r.csv").open()))
        if len(rows) != 15 or len(csv_rows) != 15:
            raise RuntimeError("expected exactly 15 unique graph-search algorithms")
        index = {r["algorithm"]: r for r in rows}
        if index["astar"]["cost"] != 8 or index["bfs"]["cost"] != 50:
            raise RuntimeError("wrong weighted example output")
        if index["hill"]["cost"] is not None:
            raise RuntimeError("non-solution must export JSON null cost")
        for command in [
            ["--algo", "unknown"], ["--backend", "bad"], ["--repeat", "0"],
            ["--start", "-1"], ["--delta", "0"], ["--depth", "-1"],
            ["--grid", str(root / "examples/maze.txt"), "--start", "9"],
            ["--algo"], ["--repeat", "2junk"],
        ]:
            r = subprocess.run([exe] + command, capture_output=True, text=True)
            if r.returncode == 0:
                raise RuntimeError("invalid CLI input accepted: " + repr(command))
        subprocess.run([exe, "--backend", "cpu", "--algo", "idastar", "--grid",
                        str(root / "examples/maze.txt"), "--queries", str(root / "examples/queries.txt"),
                        "--validate"], check=True, stdout=subprocess.DEVNULL)
    print("PASS: JSON/CSV schema, 9 rejected CLI inputs, and mixed-goal query file")


if __name__ == "__main__":
    main()
