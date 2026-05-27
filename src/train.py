import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-type",
        choices=["regression", "classification", "clustering"],
        default="regression",
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--target-column", type=str, default="sales")
    parser.add_argument("--feature-columns", type=str, default="")
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "data/raw"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    return parser.parse_args()


def resolve_csv(path):
    path = Path(path)
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in training directory: {path}")
        return csv_files[0]
    return path


def load_training_frame(train_path):
    return pd.read_csv(resolve_csv(train_path))


def parse_feature_columns(feature_columns):
    return [column.strip() for column in feature_columns.split(",") if column.strip()]


def select_features(frame, target_column, feature_columns):
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found in training data.")

    features = parse_feature_columns(feature_columns)
    if not features:
        features = [column for column in frame.columns if column != target_column]

    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns are missing from the data: {missing}")

    return frame[features], frame[target_column], features


def select_clustering_features(frame, feature_columns, target_column):
    features = parse_feature_columns(feature_columns)
    if not features:
        features = [column for column in frame.columns if column != target_column]

    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns are missing from the data: {missing}")

    return frame[features], features


def regression_metrics(y_true, predictions):
    mse = mean_squared_error(y_true, predictions)
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "mse": float(mse),
        "rmse": float(mse**0.5),
        "r2": float(r2_score(y_true, predictions)),
    }


def classification_metrics(y_true, predictions):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro")),
    }


def train_supervised(args, frame):
    x, y, feature_names = select_features(frame, args.target_column, args.feature_columns)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if args.task_type == "classification" and y.nunique() > 1 else None,
    )

    if args.task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1,
        )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = (
        classification_metrics(y_test, predictions)
        if args.task_type == "classification"
        else regression_metrics(y_test, predictions)
    )
    metrics.update(
        {
            "task_type": args.task_type,
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "target_column": args.target_column,
            "features": feature_names,
            "model": {
                "algorithm": type(model).__name__,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
            },
        }
    )
    metadata = {
        "task_type": args.task_type,
        "target_column": args.target_column,
        "feature_columns": feature_names,
    }
    return model, metadata, metrics


def train_clustering(args, frame):
    x, feature_names = select_clustering_features(frame, args.feature_columns, args.target_column)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=10)),
        ]
    )
    labels = model.fit_predict(x)
    scaled_x = model.named_steps["scaler"].transform(x)
    cluster_counts = pd.Series(labels).value_counts().sort_index().astype(int).to_dict()

    metrics = {
        "task_type": "clustering",
        "inertia": float(model.named_steps["kmeans"].inertia_),
        "silhouette_score": float(silhouette_score(scaled_x, labels)) if args.n_clusters > 1 else None,
        "n_clusters": int(args.n_clusters),
        "training_rows": int(len(frame)),
        "features": feature_names,
        "cluster_sizes": {str(cluster): count for cluster, count in cluster_counts.items()},
        "model": {
            "algorithm": "KMeans",
            "n_clusters": args.n_clusters,
            "preprocessing": "StandardScaler",
        },
    }
    metadata = {
        "task_type": "clustering",
        "feature_columns": feature_names,
        "n_clusters": args.n_clusters,
        "preprocessing": "StandardScaler",
    }
    return model, metadata, metrics


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_data_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_training_frame(args.train)
    if args.task_type == "clustering":
        model, metadata, metrics = train_clustering(args, frame)
    else:
        model, metadata, metrics = train_supervised(args, frame)

    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with open(output_dir / "metrics.txt", "w", encoding="utf-8") as metrics_file:
        for key, value in metrics.items():
            if isinstance(value, (str, int, float)) or value is None:
                metrics_file.write(f"{key}={value}\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
