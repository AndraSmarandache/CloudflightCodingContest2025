from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

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
                raise ValueError(f"Cannot parse number {token}")
            value += NUMBER_WORDS[part]
        return float(sign * value)


def load_environment(path: Path) -> Dict[int, Sequence[float]]:
    env: Dict[int, Sequence[float]] = {}
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for bop, temp, hum in reader:
            value = parse_number(temp)
            if value > 60:
                value = (value - 32) * 5.0 / 9.0
            env[int(bop)] = (float(value), float(parse_number(hum)))
    return env


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["missing"])
    df["BOP Path"] = df["BOP Path"].astype(str)
    df["Sequence"] = df["BOP Path"].apply(to_sequence)
    return df


def to_sequence(path: str) -> List[int]:
    return [int(x) for x in path.strip().split() if x]


def longest_run(values: Sequence[int]) -> int:
    best = 1
    current = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            current += 1
        else:
            best = max(best, current)
            current = 1
    return max(best, current) if values else 0


def direction_changes(values: Sequence[int]) -> int:
    if len(values) < 3:
        return 0
    diffs = [np.sign(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    return sum(1 for i in range(len(diffs) - 1) if diffs[i] != diffs[i + 1])


def backtrack_fraction(values: Sequence[int]) -> float:
    if len(values) < 3:
        return 0.0
    count = 0
    total = len(values) - 2
    for i in range(len(values) - 2):
        if values[i] == values[i + 2]:
            count += 1
    return count / total if total else 0.0


def extract_features(df: pd.DataFrame, env: Dict[int, Sequence[float]]) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        seq = row["Sequence"]
        if not seq:
            continue
        temps = [env[b][0] for b in seq]
        hums = [env[b][1] for b in seq]
        start = seq[0]
        end = seq[-1]
        uniq = len(set(seq))
        length = len(seq)
        pal = int(seq == list(reversed(seq)))
        same = Counter(seq).get(start, 0) / length
        data = {
            "flock_id": row["Flock ID"],
            "path_len": length,
            "unique_nodes": uniq,
            "unique_ratio": uniq / length,
            "start_id": start,
            "end_id": end,
            "start_end_same": int(start == end),
            "palindrome": pal,
            "mean_bop": float(np.mean(seq)),
            "std_bop": float(np.std(seq)),
            "min_bop": min(seq),
            "max_bop": max(seq),
            "mean_temp": float(np.mean(temps)),
            "std_temp": float(np.std(temps)),
            "min_temp": min(temps),
            "max_temp": max(temps),
            "mean_hum": float(np.mean(hums)),
            "std_hum": float(np.std(hums)),
            "min_hum": min(hums),
            "max_hum": max(hums),
            "first_temp": temps[0],
            "last_temp": temps[-1],
            "first_hum": hums[0],
            "last_hum": hums[-1],
            "return_ratio": same,
            "direction_changes": direction_changes(seq),
            "backtrack_frac": backtrack_fraction(seq),
            "max_run": longest_run(seq),
            "loop_ratio": (length - uniq) / length,
        }
        records.append(data)
    features = pd.DataFrame.from_records(records)
    features["Species"] = df["Species"].values[: len(features)]
    return features


def train_model(features: pd.DataFrame) -> HistGradientBoostingClassifier:
    labeled = features.dropna(subset=["Species"])
    X = labeled.drop(columns=["Species"])
    y = labeled["Species"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    model = HistGradientBoostingClassifier(
        max_iter=700,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=25,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = f1_score(y_val, preds, average="macro")
    print(f"Validation F1 Macro: {score:.3f}")
    model.fit(X, y)
    return model


def aggregate_predictions(features: pd.DataFrame, preds: Sequence[str]) -> pd.DataFrame:
    temp = features[features["Species"].isna()].copy()
    temp = temp.iloc[: len(preds)].copy()
    temp["Predicted"] = preds
    agg = temp.groupby("flock_id")["Predicted"].agg(lambda x: x.value_counts().idxmax()).reset_index()
    agg = agg.sort_values("flock_id")
    agg.columns = ["Flock ID", "Species"]
    return agg


def longest_common_prefix(paths: Sequence[List[int]]) -> int:
    base = paths[0]
    res = len(base)
    for path in paths[1:]:
        i = 0
        while i < len(base) and i < len(path) and base[i] == path[i]:
            i += 1
        res = min(res, i)
        if res == 0:
            break
    return res


def longest_common_suffix(paths: Sequence[List[int]]) -> int:
    reversed_paths = [list(reversed(path)) for path in paths]
    return longest_common_prefix(reversed_paths)


def shared_node_ratio(paths: Sequence[List[int]]) -> float:
    sets = [set(path) for path in paths]
    if not sets:
        return 0.0
    union = set.union(*sets)
    if not union:
        return 0.0
    inter = set.intersection(*sets)
    return len(inter) / len(union)


def loop_ratio(paths: Sequence[List[int]]) -> float:
    total = sum(len(path) for path in paths)
    if total == 0:
        return 0.0
    uniques = len({node for path in paths for node in path})
    return (total - uniques) / total


def pal_ratio(paths: Sequence[List[int]]) -> float:
    if not paths:
        return 0.0
    return sum(1 for path in paths if path == path[::-1]) / len(paths)


def identical_ratio(paths: Sequence[List[int]]) -> float:
    if not paths:
        return 0.0
    base = tuple(paths[0])
    matches = sum(1 for path in paths if tuple(path) == base)
    return matches / len(paths)


def backtrack_ratio(paths: Sequence[List[int]]) -> float:
    total = sum(max(0, len(path) - 2) for path in paths)
    if total == 0:
        return 0.0
    zigzags = sum(
        1
        for path in paths
        for i in range(len(path) - 2)
        if path[i] == path[i + 2]
    )
    return zigzags / total


def compute_flock_stats(group: pd.DataFrame, env: Dict[int, Sequence[float]]) -> Dict[str, float]:
    paths = group["Sequence"].tolist()
    temps = [env[b][0] for path in paths for b in path]
    hums = [env[b][1] for path in paths for b in path]
    total_len = sum(len(path) for path in paths)
    avg_len = total_len / len(paths) if paths else 0.0
    unique_nodes = len({node for path in paths for node in path})
    stats = {
        "avg_temp": float(np.mean(temps)) if temps else 0.0,
        "avg_humidity": float(np.mean(hums)) if hums else 0.0,
        "avg_len": avg_len,
        "unique_nodes": unique_nodes,
        "pal_ratio": pal_ratio(paths),
        "identical_ratio": identical_ratio(paths),
        "loop_ratio": loop_ratio(paths),
        "shared_prefix": longest_common_prefix(paths),
        "shared_suffix": longest_common_suffix(paths),
        "shared_node_ratio": shared_node_ratio(paths),
        "backtrack_ratio": backtrack_ratio(paths),
    }
    return stats


def heuristic_species(stats: Dict[str, float]) -> str:
    if stats["avg_temp"] >= 35.0:
        return "Hurracurra Bird"
    if stats["pal_ratio"] >= 0.7:
        return "Medieval Bluetit"
    if stats["identical_ratio"] >= 0.6 or stats["loop_ratio"] >= 0.5:
        return "Flanking Blackfinch"
    if stats["shared_prefix"] >= max(2, stats["avg_len"] * 0.4) and stats["shared_suffix"] <= 2:
        return "Rusty Goldhammer"
    if (
        stats["shared_node_ratio"] <= 0.15
        and stats["shared_prefix"] <= 2
        and stats["shared_suffix"] <= 2
        and stats["loop_ratio"] <= 0.3
    ):
        return "Red Firefinch"
    return "Sticky Wolfthroat"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="level_4")
    parser.add_argument("--input-file", default="level_4.in")
    parser.add_argument("--output-file", default="level_4.out")
    args = parser.parse_args(list(argv) if argv is not None else None)
    base = Path(args.base_dir)
    env = load_environment(base / "all_data_from_level_1.in")
    df = load_dataset(base / args.input_file)
    labeled_species = set(df["Species"].dropna().unique())
    if len(labeled_species) >= 3:
        features = extract_features(df, env)
        model = train_model(features)
        missing = features[features["Species"].isna()].drop(columns=["Species"])
        if missing.empty:
            print("No missing species found.")
            return
        predictions = model.predict(missing)
        agg = aggregate_predictions(features, predictions)
    else:
        groups = df.groupby("Flock ID")
        rows = []
        for flock_id, group in groups:
            has_missing = group["Species"].isna().any()
            if not has_missing:
                continue
            known = group["Species"].dropna().unique()
            if len(known) == 1:
                rows.append([flock_id, known[0]])
                continue
            stats = compute_flock_stats(group, env)
            guess = heuristic_species(stats)
            rows.append([flock_id, guess])
        agg = pd.DataFrame(rows, columns=["Flock ID", "Species"]).sort_values("Flock ID")
    output_path = base / args.output_file
    agg.to_csv(output_path, index=False)
    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()

