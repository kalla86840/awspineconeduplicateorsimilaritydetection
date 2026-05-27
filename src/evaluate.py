import argparse
import json
import os
import tarfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", type=str, default="/opt/ml/processing/model/model.tar.gz")
    parser.add_argument("--test-data", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    parser.add_argument(
        "--task-type",
        choices=["regression", "classification", "clustering"],
        default="regression",
    )
    parser.add_argument("--target-column", type=str, default="sales")
    parser.add_argument("--feature-columns", type=str, default="")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def parse_feature_columns(feature_columns):
    return [column.strip() for column in feature_columns.split(",") if column.strip()]


def load_model_bundle(model_artifact_path):
    extract_dir = Path("/tmp/model")
    extract_dir.mkdir(parents=True, exist_ok=True)

    if os.path.isdir(model_artifact_path):
        model_artifact_path = os.path.join(model_artifact_path, "model.tar.gz")

    with tarfile.open(model_artifact_path) as tar:
        tar.extractall(extract_dir)

    metadata_path = extract_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return joblib.load(extract_dir / "model.joblib"), metadata


def load_evaluation_frame(test_data_path):
    path = Path(test_data_path)
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in evaluation directory: {path}")
        path = csv_files[0]

    return pd.read_csv(path)


def resolve_features(frame, feature_columns, target_column):
    features = parse_feature_columns(feature_columns)
    if not features:
        features = [column for column in frame.columns if column != target_column]
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns are missing from the data: {missing}")
    return features


def supervised_report(model, frame, args, metadata):
    target_column = metadata.get("target_column", args.target_column)
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found in evaluation data.")

    feature_columns = metadata.get("feature_columns") or resolve_features(
        frame,
        args.feature_columns,
        target_column,
    )
    x = frame[feature_columns]
    y = frame[target_column]
    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y if args.task_type == "classification" and y.nunique() > 1 else None,
    )

    predictions = model.predict(x_test)
    if args.task_type == "classification":
        return {
            "classification_metrics": {
                "accuracy": {"value": float(accuracy_score(y_test, predictions))},
                "f1_macro": {"value": float(f1_score(y_test, predictions, average="macro"))},
            }
        }

    return {
        "regression_metrics": {
            "mae": {"value": float(mean_absolute_error(y_test, predictions))},
            "mse": {"value": float(mean_squared_error(y_test, predictions))},
            "r2": {"value": float(r2_score(y_test, predictions))},
        }
    }


def clustering_report(model, frame, args, metadata):
    feature_columns = metadata.get("feature_columns") or resolve_features(
        frame,
        args.feature_columns,
        args.target_column,
    )
    x = frame[feature_columns]
    labels = model.predict(x)
    scaled_x = model.named_steps["scaler"].transform(x) if hasattr(model, "named_steps") else x
    n_clusters = int(metadata.get("n_clusters", len(set(labels))))
    return {
        "clustering_metrics": {
            "inertia": {"value": float(model.named_steps["kmeans"].inertia_)},
            "silhouette_score": {
                "value": float(silhouette_score(scaled_x, labels)) if n_clusters > 1 else None
            },
            "n_clusters": {"value": n_clusters},
        }
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, metadata = load_model_bundle(args.model_artifact)
    task_type = metadata.get("task_type", args.task_type)
    args.task_type = task_type
    frame = load_evaluation_frame(args.test_data)

    if task_type == "clustering":
        report = clustering_report(model, frame, args, metadata)
    else:
        report = supervised_report(model, frame, args, metadata)

    with open(os.path.join(args.output_dir, "evaluation.json"), "w", encoding="utf-8") as output_file:
        json.dump(report, output_file)


if __name__ == "__main__":
    main()
