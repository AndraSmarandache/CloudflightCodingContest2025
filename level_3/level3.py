from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

NUMBER_WORDS: Dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

SPECIES = [
    "Hurracurra Bird",
    "Medieval Bluetit",
    "Flanking Blackfinch",
    "Rusty Goldhammer",
    "Red Firefinch",
    "Sticky Wolfthroat",
]


def parse_number(token: str) -> float:
    token = token.strip().lower().replace("-", " ")
    try:
        return float(token)
    except ValueError:
        sign = 1
        if token.startswith("minus "):
            sign = -1
            token = token.replace("minus ", "", 1)
        parts = token.split()
        value = 0
        for part in parts:
            if part not in NUMBER_WORDS:
                raise ValueError(f"Cannot parse number: {token}")
            value += NUMBER_WORDS[part]
        return float(sign * value)


def load_environment(path: Path) -> Dict[int, float]:
    env: Dict[int, float] = {}
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for bop, temp, _ in reader:
            value = parse_number(temp)
            if value > 60:
                value = (value - 32) * 5.0 / 9.0
            env[int(bop)] = float(value)
    return env


def read_dataset(path: Path) -> Dict[int, List[List[int]]]:
    flocks: Dict[int, List[List[int]]] = {}
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for flock, seq in reader:
            flocks.setdefault(int(flock), []).append([int(x) for x in seq.split()])
    return flocks


def longest_common_prefix(paths: Sequence[List[int]]) -> int:
    prefix = list(paths[0])
    for path in paths[1:]:
        i = 0
        while i < len(prefix) and i < len(path) and prefix[i] == path[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return len(prefix)


def compute_features(paths: Sequence[List[int]], env: Dict[int, float]) -> Dict[str, float]:
    pal = all(p == p[::-1] for p in paths)
    same = len({tuple(p) for p in paths}) == 1
    temps = [env[b] for p in paths for b in p]
    avg_temp = sum(temps) / len(temps)
    avg_len = sum(len(p) for p in paths) / len(paths)
    node_set = {b for p in paths for b in p}
    lcp = longest_common_prefix(paths)
    return {
        "pal": pal,
        "same": same,
        "avg_temp": avg_temp,
        "avg_len": avg_len,
        "nodes": node_set,
        "lcp": lcp,
    }


def classify(flocks: Dict[int, Dict[str, float]]) -> Dict[int, str]:
    assignments: Dict[int, str] = {}
    pal_ids = [fid for fid, feat in flocks.items() if feat["pal"]]
    pal_ids.sort(key=lambda fid: len(flocks[fid]["nodes"]))
    if len(pal_ids) == 2:
        wolf, blue = pal_ids
        assignments[wolf] = "Sticky Wolfthroat"
        assignments[blue] = "Medieval Bluetit"
    flanking_candidates = [fid for fid, feat in flocks.items() if not feat["pal"] and feat["same"]]
    if flanking_candidates:
        assignments[flanking_candidates[0]] = "Flanking Blackfinch"
    hurracurra = max(flocks, key=lambda fid: flocks[fid]["avg_temp"])
    assignments[hurracurra] = "Hurracurra Bird"
    rusty_candidates = [
        fid
        for fid, feat in flocks.items()
        if fid not in assignments and not feat["pal"] and feat["lcp"] > 1
    ]
    if rusty_candidates:
        assignments[rusty_candidates[0]] = "Rusty Goldhammer"
    for fid in flocks:
        if fid not in assignments:
            assignments[fid] = "Red Firefinch"
    return assignments


def solve(path: Path, env: Dict[int, float]) -> Dict[int, str]:
    data = read_dataset(path)
    features = {fid: compute_features(paths, env) for fid, paths in data.items()}
    result = classify(features)
    return result


def write_output(path: Path, mapping: Dict[int, str]) -> None:
    rows = [["Flock ID", "Species"]]
    for fid in sorted(mapping):
        rows.append([str(fid), mapping[fid]])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Dataset files to solve, defaults to level_3_a.in level_3_b.in level_3_c.in",
    )
    parser.add_argument("--base-dir", default="level_3")
    args = parser.parse_args(list(argv) if argv is not None else None)
    base = Path(args.base_dir)
    env = load_environment(base / "all_data_from_level_1.in")
    files = [base / name for name in args.inputs] if args.inputs else [
        base / "level_3_a.in",
        base / "level_3_b.in",
        base / "level_3_c.in",
    ]
    for file in files:
        mapping = solve(file, env)
        write_output(file.with_suffix(".out"), mapping)


if __name__ == "__main__":
    main()

