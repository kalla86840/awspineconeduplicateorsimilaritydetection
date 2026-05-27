import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
    model_path = Path(model_dir) / "model.joblib"
    metadata_path = Path(model_dir) / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "model": joblib.load(model_path),
        "metadata": metadata,
    }


def input_fn(request_body, request_content_type):
    content_type = request_content_type.split(";")[0].strip().lower()
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)
    instances = payload.get("instances")
    if instances is None:
        raise ValueError("Request JSON must include an 'instances' field.")

    return instances


def _as_model_input(input_data, feature_columns):
    if input_data and isinstance(input_data[0], dict):
        frame = pd.DataFrame(input_data)
        if feature_columns:
            missing = [column for column in feature_columns if column not in frame.columns]
            if missing:
                raise ValueError(f"Request instances are missing feature columns: {missing}")
            return frame[feature_columns]
        return frame
    return np.asarray(input_data)


def predict_fn(input_data, model_bundle):
    if isinstance(model_bundle, dict) and "model" in model_bundle:
        model = model_bundle["model"]
        metadata = model_bundle.get("metadata", {})
    else:
        model = model_bundle
        metadata = {}

    model_input = _as_model_input(input_data, metadata.get("feature_columns", []))
    predictions = model.predict(model_input)

    if metadata.get("task_type") == "clustering":
        distances = model.transform(model_input).min(axis=1)
        return [
            {"cluster": int(cluster), "distance_to_centroid": float(distance)}
            for cluster, distance in zip(predictions, distances)
        ]

    return predictions


def output_fn(prediction, response_content_type):
    if hasattr(prediction, "tolist"):
        prediction = prediction.tolist()
    body = json.dumps({"predictions": prediction})
    return body, "application/json"
