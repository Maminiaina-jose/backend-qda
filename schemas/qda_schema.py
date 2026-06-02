from pydantic import BaseModel
from typing import Any


class ModelInfo(BaseModel):
    features: list[str]
    classes: list[str]
    priors: list[float]
    means: list[list[float]]
    covariances: list[list[list[float]]]
    scaler_mean: list[float]
    scaler_scale: list[float]


class PredictResponse(BaseModel):
    result_id: str
    data: list[dict[str, Any]]
    predictions: list[str]
    classes: list[str]
    n_rows: int
    model_info: ModelInfo


class ResultResponse(BaseModel):
    result_id: str
    data: list[dict[str, Any]]
    predictions: list[str]
    classes: list[str]
    n_rows: int
    model_info: ModelInfo
