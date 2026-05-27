import json
from pathlib import Path

from src import inference, train


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "cars_1020.csv"


def run_training(tmp_path, **overrides):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    defaults = {
        "task_type": "regression",
        "n_estimators": 20,
        "max_depth": None,
        "n_clusters": 4,
        "random_state": 42,
        "test_size": 0.2,
        "target_column": "sales",
        "feature_columns": "age,gender,miles,debt,income",
        "train": str(DATA_PATH),
        "model_dir": str(model_dir),
        "output_data_dir": str(output_dir),
    }
    defaults.update(overrides)

    original_parse_args = train.parse_args
    train.parse_args = lambda: type("Args", (), defaults)()
    try:
        train.main()
    finally:
        train.parse_args = original_parse_args

    return model_dir, output_dir


def test_classification_training_writes_metrics_and_metadata(tmp_path):
    model_dir, output_dir = run_training(
        tmp_path,
        task_type="classification",
        target_column="gender",
        feature_columns="age,miles,debt,income,sales",
    )

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert metadata["task_type"] == "classification"
    assert metadata["target_column"] == "gender"
    assert metrics["model"]["algorithm"] == "RandomForestClassifier"
    assert {"accuracy", "f1_macro"}.issubset(metrics)

    model_bundle = inference.model_fn(str(model_dir))
    parsed = inference.input_fn(
        json.dumps({"instances": [{"age": 28, "miles": 23, "debt": 0, "income": 4099, "sales": 620}]}),
        "application/json",
    )
    prediction = inference.predict_fn(parsed, model_bundle)
    assert prediction.tolist()[0] in {0, 1}


def test_clustering_training_returns_cluster_predictions(tmp_path):
    model_dir, output_dir = run_training(
        tmp_path,
        task_type="clustering",
        feature_columns="age,miles,debt,income,sales",
        n_clusters=4,
    )

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert metadata["task_type"] == "clustering"
    assert metadata["n_clusters"] == 4
    assert metrics["model"]["algorithm"] == "KMeans"
    assert "silhouette_score" in metrics

    model_bundle = inference.model_fn(str(model_dir))
    parsed = inference.input_fn(
        json.dumps({"instances": [{"age": 28, "miles": 23, "debt": 0, "income": 4099, "sales": 620}]}),
        "application/json",
    )
    prediction = inference.predict_fn(parsed, model_bundle)
    assert prediction[0]["cluster"] in {0, 1, 2, 3}
    assert prediction[0]["distance_to_centroid"] >= 0
