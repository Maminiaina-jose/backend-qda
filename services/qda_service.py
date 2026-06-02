import joblib
import numpy as np
import pandas as pd
import os
import uuid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from config import settings
BASE_DIR = settings.SKLEARN_MODEL_DIR
os.makedirs(BASE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.npy")
ACCURACY_PATH = os.path.join(BASE_DIR, "accuracy.npy")

_results_store: dict[str, dict] = {}


def _read_csv_auto(csv_path: str) -> tuple[pd.DataFrame, str | None, list[str]]:
    df = pd.read_csv(csv_path)
    try:
        float(df.columns[0])
        df = pd.read_csv(csv_path, header=None)
        n = len(df.columns)
        df.columns = [f"feat_{i}" for i in range(n)]
        target_col = df.columns[0]
        feature_cols = df.columns[1:].tolist()
    except ValueError:
        target_col = df.columns[-1] if _is_categorical(df, df.columns[-1]) else None
        feature_cols = df.columns[:-1].tolist() if target_col else df.columns.tolist()
    return df, target_col, feature_cols


def _is_categorical(df: pd.DataFrame, col: str) -> bool:
    if df[col].dtype in ("object", "category", "bool"):
        return True
    n = len(df)
    if n == 0:
        return False
    return df[col].nunique() <= 10


def _is_labeled(df: pd.DataFrame, target_col: str, threshold: float = 0.3) -> bool:
    n = len(df)
    if n == 0:
        return False
    unique = df[target_col].nunique()
    return unique / n < threshold or unique <= 10


def train_model(csv_path: str) -> dict:
    df, target_col, feature_cols = _read_csv_auto(csv_path)
    if target_col is None:
        raise ValueError("No target column detected. CSV must have a label column.")

    X = df[feature_cols].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(X_scaled, y)

    joblib.dump(qda, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    np.save(FEATURES_PATH, np.array(feature_cols))

    accuracy = qda.score(X_scaled, y)
    np.save(ACCURACY_PATH, np.array(accuracy))

    return {
        "accuracy": float(accuracy),
        "classes": qda.classes_.tolist(),
        "features": feature_cols,
        "n_features": len(feature_cols),
        "auto_trained": False,
    }


def predict_csv(csv_path: str) -> dict:
    df, target_col, feature_cols = _read_csv_auto(csv_path)

    has_labels = target_col is not None
    model_exists = os.path.exists(MODEL_PATH)
    should_retrain = False

    if has_labels and model_exists:
        try:
            existing_features = np.load(FEATURES_PATH, allow_pickle=True).tolist()
            if set(feature_cols) != set(existing_features) or len(feature_cols) != len(existing_features):
                should_retrain = True
        except Exception:
            should_retrain = True
    elif has_labels and not model_exists:
        should_retrain = True

    if should_retrain:
        train_model(csv_path)
        model_exists = True

    if not model_exists:
        raise FileNotFoundError(
            "No model available. Upload a labeled CSV (last col = class) to auto-train."
        )

    qda: QuadraticDiscriminantAnalysis = joblib.load(MODEL_PATH)
    scaler: StandardScaler = joblib.load(SCALER_PATH)
    feature_names: list[str] = np.load(FEATURES_PATH, allow_pickle=True).tolist()

    matched_cols = [c for c in feature_names if c in df.columns.tolist()]
    if len(matched_cols) == len(feature_names):
        X = df[matched_cols].values
    elif len(df.columns) >= len(feature_names):
        X = df.iloc[:, :len(feature_names)].values.astype(np.float64)
    else:
        raise ValueError(
            f"Column mismatch. Model expects {len(feature_names)} features: {feature_names}. "
            f"Your file has {len(df.columns)} columns: {df.columns.tolist()}"
        )

    X = X.astype(np.float64)
    X_scaled = scaler.transform(X)

    predictions = qda.predict(X_scaled)
    probabilities = qda.predict_proba(X_scaled)

    result_df = df.copy()
    result_df["prediction"] = predictions
    for i, cls in enumerate(qda.classes_):
        result_df[f"prob_{cls}"] = probabilities[:, i]

    model_params = _extract_model_params(qda, scaler, feature_names)

    result = {
        "data": result_df.to_dict(orient="records"),
        "predictions": predictions.tolist(),
        "classes": qda.classes_.tolist(),
        "n_rows": len(predictions),
        "model_info": model_params,
    }

    result_id = str(uuid.uuid4())
    _results_store[result_id] = result

    return {**result, "result_id": result_id, "auto_trained": should_retrain}


def _extract_model_params(
    qda: QuadraticDiscriminantAnalysis,
    scaler: StandardScaler,
    feature_names: list[str],
) -> dict:
    try:
        accuracy = float(np.load(ACCURACY_PATH))
    except Exception:
        accuracy = None
    return {
        "model_type": "sklearn",
        "features": feature_names,
        "classes": qda.classes_.tolist(),
        "priors": qda.priors_.tolist(),
        "means": [m.tolist() for m in qda.means_],
        "covariances": [c.tolist() for c in qda.covariance_],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "accuracy": accuracy,
    }


def get_result(result_id: str) -> dict | None:
    return _results_store.get(result_id)


def get_model_info() -> dict | None:
    if not os.path.exists(MODEL_PATH):
        return None
    qda: QuadraticDiscriminantAnalysis = joblib.load(MODEL_PATH)
    scaler: StandardScaler = joblib.load(SCALER_PATH)
    feature_names: list[str] = np.load(FEATURES_PATH, allow_pickle=True).tolist()
    return _extract_model_params(qda, scaler, feature_names)
