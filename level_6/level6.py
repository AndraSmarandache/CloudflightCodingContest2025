from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def load_day(df: pd.DataFrame, day: int) -> pd.Series:
    return (
        df[df["Day"] == day]
        .groupby("BOP")["Arrivals"]
        .sum()
        .sort_values(ascending=False)
    )


def verify_time_reversal(level5: pd.DataFrame, level6: pd.DataFrame) -> None:
    arr5 = (
        level5.pivot(index="Day", columns="BOP", values="Arrivals")
        .loc[range(760, 730, -1)]
        .to_numpy()
    )
    arr6 = (
        level6.pivot(index="Day", columns="BOP", values="Arrivals")
        .loc[range(761, 791)]
        .to_numpy()
    )
    if not np.allclose(arr6, arr5):
        raise ValueError("New weather data is not the reverse of the previous 30 days.")


def top50_day730(level5_path: Path) -> List[int]:
    df = pd.read_csv(level5_path, na_values=["missing"])
    day730 = load_day(df, 730)
    return day730.head(50).index.tolist()


def write_output(rows: List[Tuple[int, List[int]]], destination: Path) -> None:
    lines = ["Day,Top 50 Arrivals BOPs"]
    for day, bop_list in rows:
        lines.append(f"{day}," + " ".join(str(bop) for bop in bop_list))
    destination.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict day 791 top BOPs.")
    parser.add_argument("--base-dir", default="level_6")
    parser.add_argument("--output-file", default="level_6.out")
    parser.add_argument("--verify", action="store_true", help="Check the reversal pattern.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    base = Path(args.base_dir)
    level5_path = base / "all_data_from_level_5.in"
    level6_path = base / "level_6.in"
    if args.verify:
        verify_time_reversal(
            pd.read_csv(level5_path, na_values=["missing"]),
            pd.read_csv(level6_path, na_values=["missing"]),
        )
    top50 = top50_day730(level5_path)
    write_output([(791, top50)], base / args.output_file)


if __name__ == "__main__":
    main()

