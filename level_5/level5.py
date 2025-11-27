from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, na_values=["missing"])
    df = df.rename(columns={"Day": "day", "BOP": "bop", "Arrivals": "arrivals"})
    days = np.arange(1, df["day"].max() + 1, dtype=np.int32)
    bops = np.sort(df["bop"].unique())
    arrivals = (
        df.pivot(index="day", columns="bop", values="arrivals")
        .reindex(index=days, columns=bops)
        .to_numpy(dtype=np.float32)
    )
    occupancy = (
        df.pivot(index="day", columns="bop", values="Occupancy")
        .reindex(index=days, columns=bops)
        .to_numpy(dtype=np.float32)
    )
    return days, bops, arrivals, occupancy


def evaluate_alpha(arrivals: np.ndarray, occupancy: np.ndarray, start_day: int, end_day: int, alpha: float) -> float:
    total = 0.0
    days_count = end_day - start_day + 1
    for day in range(start_day, end_day + 1):
        idx = day - 1
        prev_idx = idx - 1
        ratio = occupancy[idx] / occupancy[prev_idx]
        ratio = np.clip(ratio, 0.1, 10.0)
        preds = arrivals[prev_idx] * (ratio ** alpha)
        actual_top = set(np.argsort(arrivals[idx])[::-1][:50])
        pred_top = set(np.argsort(preds)[::-1][:50])
        total += len(actual_top & pred_top) / 50.0
    return total / days_count


def select_alpha(arrivals: np.ndarray, occupancy: np.ndarray) -> Tuple[float, float]:
    best_alpha = 0.0
    best_score = -1.0
    for alpha in np.linspace(0.0, 1.0, 11):
        score = evaluate_alpha(arrivals, occupancy, 701, 730, alpha)
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha, best_score


def persistence_forecast(
    arrivals: np.ndarray,
    occupancy: np.ndarray,
    start_day_idx: int,
    bop_ids: np.ndarray,
    alpha: float,
) -> List[Tuple[int, List[int]]]:
    results: List[Tuple[int, List[int]]] = []
    for idx in range(start_day_idx, arrivals.shape[0]):
        ratio = occupancy[idx] / occupancy[idx - 1]
        ratio = np.clip(ratio, 0.1, 10.0)
        predictions = arrivals[idx - 1] * (ratio ** alpha)
        arrivals[idx] = predictions
        order = np.argsort(predictions)[::-1][:50]
        top_bops = bop_ids[order].tolist()
        results.append((idx + 1, top_bops))
    return results


def write_output(rows: List[Tuple[int, List[int]]], path: Path) -> None:
    lines = ["Day,Top 50 Arrivals BOPs"]
    for day, bop_list in rows:
        line = f"{day}," + " ".join(str(bop) for bop in bop_list)
        lines.append(line)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="level_5")
    parser.add_argument("--input-file", default="level_5.in")
    parser.add_argument("--output-file", default="level_5.out")
    args = parser.parse_args(list(argv) if argv is not None else None)
    base = Path(args.base_dir)
    days, bop_ids, arrivals, occupancy = load_dataset(base / args.input_file)
    train_days = 730
    alpha, score = select_alpha(arrivals.copy(), occupancy.copy())
    print(f"[info] Selected alpha={alpha:.2f} (validation accuracy {score:.3f})")
    predictions = persistence_forecast(arrivals, occupancy, train_days, bop_ids, alpha)
    write_output(predictions, base / args.output_file)


if __name__ == "__main__":
    main()

