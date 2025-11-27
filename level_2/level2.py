from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

WORD_TO_NUMBER: Dict[str, int] = {
    "minus": -1,
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

LEVEL2_DEFAULT_FILES = ["level_2_a.in", "level_2_b.in", "level_2_c.in"]


def parse_numeric_token(token: str) -> float:
    token = str(token).strip()
    if not token:
        raise ValueError("Empty numeric token.")

    try:
        return float(token)
    except ValueError:
        normalized = token.lower().replace("-", " ")
        sign = 1
        if normalized.startswith("minus "):
            sign = -1
            normalized = normalized.replace("minus ", "", 1)
        parts = [WORD_TO_NUMBER.get(part) for part in normalized.split()]
        if all(part is not None for part in parts) and parts:
            base = parts[0]
            if len(parts) > 1:
                base += sum(parts[1:])
            return float(sign * base)
        raise ValueError(f"Cannot parse numeric token '{token}'.")


def load_environment_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Temperature [°C]"] = df["Temperature [°C]"].apply(parse_numeric_token).astype(
        float
    )
    df["Humidity [%]"] = df["Humidity [%]"].apply(parse_numeric_token).astype(float)
    mask_fahrenheit = df["Temperature [°C]"] > 60
    df.loc[mask_fahrenheit, "Temperature [°C]"] = (
        (df.loc[mask_fahrenheit, "Temperature [°C]"] - 32) * 5.0 / 9.0
    )
    return df.rename(
        columns={
            "BOP": "bop",
            "Temperature [°C]": "temperature_c",
            "Humidity [%]": "humidity",
        }
    )


def load_level2_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["missing"])
    df = df.rename(
        columns={
            "BOP": "bop",
            "Vegetation [%]": "vegetation",
            "Insects [g/m²]": "insects",
            "Urban Light [%]": "urban_light",
            "Bird Love Score [<3]": "bird_love_score",
        }
    )
    df["bop"] = df["bop"].astype(int)
    numeric_cols = ["vegetation", "insects", "urban_light"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["bird_love_score"] = pd.to_numeric(df["bird_love_score"], errors="coerce")
    df["source_file"] = path.name
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    eps = 1.0
    engineered["veg_insects"] = engineered["vegetation"] * engineered["insects"]
    engineered["veg_light"] = engineered["vegetation"] * engineered["urban_light"]
    engineered["insects_light"] = engineered["insects"] * engineered["urban_light"]
    engineered["veg_ratio"] = engineered["vegetation"] / (engineered["urban_light"] + eps)
    engineered["insects_ratio"] = engineered["insects"] / (
        engineered["urban_light"] + eps
    )
    engineered["temp_humidity"] = engineered["temperature_c"] * engineered["humidity"]
    engineered["temp_light"] = engineered["temperature_c"] * engineered["urban_light"]
    engineered["humidity_light"] = engineered["humidity"] * engineered["urban_light"]
    engineered["veg_sq"] = engineered["vegetation"] ** 2
    engineered["insects_sq"] = engineered["insects"] ** 2
    engineered["light_sq"] = engineered["urban_light"] ** 2
    engineered["temp_sq"] = engineered["temperature_c"] ** 2
    engineered["humidity_sq"] = engineered["humidity"] ** 2
    return engineered


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "vegetation",
        "insects",
        "urban_light",
        "temperature_c",
        "humidity",
        "veg_insects",
        "veg_light",
        "insects_light",
        "veg_ratio",
        "insects_ratio",
        "temp_humidity",
        "temp_light",
        "humidity_light",
        "veg_sq",
        "insects_sq",
        "light_sq",
        "temp_sq",
        "humidity_sq",
    ]
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered feature columns: {missing}")
    return df[features]


def train_regressor(data: pd.DataFrame) -> HistGradientBoostingRegressor:
    data = engineer_features(data)
    feature_matrix = build_design_matrix(data)
    target = data["bird_love_score"].to_numpy()

    if len(data) > 200:
        X_train, X_val, y_train, y_val = train_test_split(
            feature_matrix, target, test_size=0.15, random_state=42
        )
        interim_model = HistGradientBoostingRegressor(
            max_depth=8,
            max_iter=700,
            learning_rate=0.04,
            min_samples_leaf=15,
            l2_regularization=0.2,
            random_state=42,
        )
        interim_model.fit(X_train, y_train)
        val_pred = interim_model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        print(f"[info] Validation RMSE on holdout: {rmse:.3f}")

    model = HistGradientBoostingRegressor(
        max_depth=8,
        max_iter=800,
        learning_rate=0.04,
        min_samples_leaf=10,
        l2_regularization=0.15,
        random_state=42,
    )
    model.fit(feature_matrix, target)
    return model


def predict_missing(model: HistGradientBoostingRegressor, df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(df)
    feature_matrix = build_design_matrix(df)
    predictions = model.predict(feature_matrix)
    output = df[["bop"]].copy()
    output["Bird Love Score [<3]"] = predictions
    return output


def format_and_write_predictions(predictions: pd.DataFrame, destination: Path) -> None:
    predictions = predictions.copy()
    predictions = predictions.rename(columns={"bop": "BOP"})
    predictions["Bird Love Score [<3]"] = predictions["Bird Love Score [<3]"].round(2)
    predictions.to_csv(destination, index=False)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Predict Bird Love Scores for missing BOP entries."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
    )
    parser.add_argument(
        "--base-dir",
        default="level_2",
        help="Directory containing level 2 data files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.base_dir)
    env_path = base_dir / "all_data_from_level_1.in"
    environment_df = load_environment_data(env_path)

    all_level2_paths = sorted(base_dir.glob("level_2_*.in"))
    if not all_level2_paths:
        raise SystemExit(f"No level_2_*.in files found inside {base_dir}.")

    level2_frames = [load_level2_file(path) for path in all_level2_paths]
    combined = (
        pd.concat(level2_frames, ignore_index=True)
        .merge(environment_df, on="bop", how="left", validate="m:1")
    )
    combined = combined.dropna(
        subset=["vegetation", "insects", "urban_light", "temperature_c", "humidity"]
    )
    labeled = combined.dropna(subset=["bird_love_score"]).reset_index(drop=True)
    if labeled.empty:
        raise SystemExit("No labeled rows found to train the model.")

    model = train_regressor(labeled)

    if args.inputs:
        target_files = [base_dir / Path(name) for name in args.inputs]
    else:
        target_files = [base_dir / name for name in LEVEL2_DEFAULT_FILES]

    for path in target_files:
        if not path.exists():
            print(f"[warn] Skipping missing file {path}")
            continue
        df = load_level2_file(path).merge(
            environment_df, on="bop", how="left", validate="m:1"
        )
        missing_mask = df["bird_love_score"].isna()
        if not missing_mask.any():
            print(f"[info] No missing Bird Love Scores in {path.name}.")
            continue
        to_predict = df.loc[missing_mask].copy()
        predictions = predict_missing(model, to_predict)
        destination = path.with_suffix(".out")
        format_and_write_predictions(predictions, destination)
        print(f"[info] Wrote predictions for {path.name} -> {destination.name}")


if __name__ == "__main__":
    main()