from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os

from services.qda_service import (
    train_model as sklearn_train,
    predict_csv as sklearn_predict,
    get_model_info as sklearn_info,
    get_result as get_any_result,
)
from services.neural_service import (
    train_model as neural_train,
    predict_csv as neural_predict,
    get_model_info as neural_info,
)

router = APIRouter(prefix="/qda", tags=["QDA"])


def _save_upload(file: UploadFile) -> str:
    os.makedirs("data", exist_ok=True)
    path = f"data/{file.filename or 'upload.csv'}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return path


# ---------------------------------------------------------------------------
# Sklearn QDA
# ---------------------------------------------------------------------------

@router.post("/sklearn/train")
def train_sklearn(file: UploadFile = File(...)):
    try:
        r = sklearn_train(_save_upload(file))
        return {
            "message": f"Sklearn QDA entraîné sur {r['n_features']} features",
            "classes": r["classes"],
            "features": r["features"],
            "accuracy": r["accuracy"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sklearn/predict")
def predict_sklearn(file: UploadFile = File(...)):
    try:
        return sklearn_predict(_save_upload(file))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/sklearn/model-info")
def model_info_sklearn():
    info = sklearn_info()
    if info is None:
        raise HTTPException(
            status_code=404,
            detail="No sklearn model trained. Use POST /qda/sklearn/train first.",
        )
    return info


# ---------------------------------------------------------------------------
# Neural QDA (PyTorch)
# ---------------------------------------------------------------------------

@router.post("/neural/train")
def train_neural(file: UploadFile = File(...)):
    try:
        r = neural_train(_save_upload(file))
        return {
            "message": f"Neural QDA entraîné sur {r['n_features']} features",
            "classes": r["classes"],
            "features": r["features"],
            "accuracy": r["accuracy"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/neural/predict")
def predict_neural(file: UploadFile = File(...)):
    try:
        return neural_predict(_save_upload(file))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/neural/model-info")
def model_info_neural():
    info = neural_info()
    if info is None:
        raise HTTPException(
            status_code=404,
            detail="No neural model trained. Use POST /qda/neural/train first.",
        )
    return info


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

@router.get("/result/{result_id}")
def get_result(result_id: str):
    result = get_any_result(result_id)
    if result is None:
        from services.neural_service import get_result as neural_get
        result = neural_get(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result
