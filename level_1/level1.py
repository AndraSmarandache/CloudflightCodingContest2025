from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Dict, Iterable, List, Sequence, Tuple

WORD_TO_NUMBER: Dict[str, int] = {
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


def parse_int(token: str) -> int:
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        normalized = token.lower().replace("-", " ")
        if normalized in WORD_TO_NUMBER:
            return WORD_TO_NUMBER[normalized]
        parts = [WORD_TO_NUMBER.get(part) for part in normalized.split()]
        if all(part is not None for part in parts) and parts:
            # Handles values such as "twenty one" => 20 + 1 = 21
            return parts[0] + sum(parts[1:])
        raise ValueError(f"Cannot parse integer value from '{token}'.")


def load_bops(path: pathlib.Path) -> List[Tuple[int, int, int]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        entries = []
        for row in reader:
            if not row:
                continue
            bop_id = int(row[0].strip())
            temperature = parse_int(row[1])
            humidity = parse_int(row[2])
            entries.append((bop_id, temperature, humidity))
    return entries


def sort_bops(entries: Sequence[Tuple[int, int, int]]) -> List[int]:
    return [
        bop_id
        for bop_id, _, _ in sorted(
            entries, key=lambda item: (-item[1], item[2], item[0])
        )
    ]


def write_output(path: pathlib.Path, bop_ids: Iterable[int]) -> None:
    path.write_text(" ".join(str(bop_id) for bop_id in bop_ids))


def resolve_inputs(paths: Sequence[str]) -> List[pathlib.Path]:
    if paths:
        return [pathlib.Path(p) for p in paths]
    return sorted(pathlib.Path("level_1").glob("level_1_*.in"))


def process_file(path: pathlib.Path) -> pathlib.Path:
    entries = load_bops(path)
    sorted_ids = sort_bops(entries)
    output_path = path.with_suffix(".out")
    write_output(output_path, sorted_ids)
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sort Bird Observation Points by temperature, humidity, and id."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
    )
    args = parser.parse_args(argv)

    inputs = resolve_inputs(args.inputs)
    if not inputs:
        raise SystemExit("No input files were found.")

    for input_path in inputs:
        output_path = process_file(input_path)
        print(f"{input_path} -> {output_path}")


if __name__ == "__main__":
    main()

