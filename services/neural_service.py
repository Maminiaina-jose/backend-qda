import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
import os
import uuid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from models.qda_model import QDA_NeuralNet

from config import settings
BASE_DIR = settings.SKLEARN_MODEL_DIR
os.makedirs(BASE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.npy")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.npy")

HIDDEN_DIM = 32
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10
TEST_SIZE = 0.2

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


def train_model(csv_path: str) -> dict:
    df, target_col, feature_cols = _read_csv_auto(csv_path)
    if target_col is None:
        raise ValueError("No target column detected. CSV must have a label column.")

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    np.save(LABEL_ENCODER_PATH, le.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]
    output_dim = len(np.unique(y))

    model = QDA_NeuralNet(input_dim, HIDDEN_DIM, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    best_state = None
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_test_t).argmax(dim=1)
            acc = (pred == y_test_t).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            break

    checkpoint = {
        "model_state_dict": best_state,
        "input_dim": input_dim,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": output_dim,
        "classes": le.classes_.tolist(),
        "accuracy": best_acc,
        "feature_names": feature_cols,
    }
    torch.save(checkpoint, MODEL_PATH)

    return {
        "accuracy": best_acc,
        "classes": le.classes_.tolist(),
        "features": feature_cols,
        "n_features": len(feature_cols),
    }


def predict_csv(csv_path: str) -> dict:
    df, target_col, feature_cols = _read_csv_auto(csv_path)

    has_labels = target_col is not None
    model_exists = os.path.exists(MODEL_PATH)
    should_retrain = False

    if has_labels and model_exists:
        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            existing_features = ckpt.get("feature_names", [])
            if set(feature_cols) != set(existing_features) or len(feature_cols) != len(existing_features):
                should_retrain = True
        except Exception:
            should_retrain = True
    elif has_labels and not model_exists:
        should_retrain = True

    if should_retrain:
        train_model(csv_path)
        df, target_col, feature_cols = _read_csv_auto(csv_path)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No neural model trained. Upload labeled CSV to auto-train.")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    scaler: StandardScaler = joblib.load(SCALER_PATH)
    feature_names: list[str] = checkpoint["feature_names"]
    classes: list[str] = checkpoint["classes"]

    available_cols = df.columns.tolist()
    matched_cols = [c for c in feature_names if c in available_cols]

    if len(matched_cols) == len(feature_names):
        X = df[matched_cols].values
    else:
        try:
            float(available_cols[0])
            df = pd.read_csv(csv_path, header=None)
            df.columns = [f"feat_{i}" for i in range(len(df.columns))]
        except ValueError:
            pass
        matched_cols = [c for c in feature_names if c in df.columns.tolist()]
        if len(matched_cols) == len(feature_names):
            X = df[matched_cols].values
        elif len(df.columns) >= len(feature_names):
            X = df.iloc[:, :len(feature_names)].values
        else:
            raise ValueError(
                f"Column mismatch. Model expects {len(feature_names)} features: {feature_names}. "
                f"Your file has {len(df.columns)} columns: {df.columns.tolist()}"
            )

    X = X.astype(np.float32)
    X_scaled = scaler.transform(X)

    model = QDA_NeuralNet(
        checkpoint["input_dim"],
        checkpoint["hidden_dim"],
        checkpoint["output_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = logits.argmax(dim=1).numpy()

    predictions = [classes[p] for p in preds]

    result_df = df.copy()
    result_df["prediction"] = predictions
    for i, cls in enumerate(classes):
        result_df[f"prob_{cls}"] = probs[:, i]

    result = {
        "data": result_df.to_dict(orient="records"),
        "predictions": predictions,
        "classes": classes,
        "n_rows": len(predictions),
        "model_info": {
            "model_type": "neural",
            "features": feature_names,
            "classes": classes,
            "accuracy": checkpoint["accuracy"],
            "architecture": {
                "input_dim": checkpoint["input_dim"],
                "hidden_dim": checkpoint["hidden_dim"],
                "output_dim": checkpoint["output_dim"],
            },
        },
    }

    result_id = str(uuid.uuid4())
    _results_store[result_id] = result

    return {**result, "result_id": result_id, "auto_trained": should_retrain}


def get_result(result_id: str) -> dict | None:
    return _results_store.get(result_id)


def get_model_info() -> dict | None:
    if not os.path.exists(MODEL_PATH):
        return None
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    return {
        "model_type": "neural",
        "features": checkpoint["feature_names"],
        "classes": checkpoint["classes"],
        "accuracy": checkpoint["accuracy"],
        "architecture": {
            "input_dim": checkpoint["input_dim"],
            "hidden_dim": checkpoint["hidden_dim"],
            "output_dim": checkpoint["output_dim"],
        },
    }
