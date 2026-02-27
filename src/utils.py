import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict
import skops.io as sio

from src.exception import CustomException

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, average_precision_score,
    roc_auc_score, roc_curve,
)
    

def compute_metrics(y_true: np.ndarray, 
                    y_pred: np.ndarray, 
                    y_proba: np.ndarray
                    ) -> Dict[str, float]:
    """
    Compute a standard set of binary classification metrics.

    Parameters
    ----------
    y_true  : ground-truth labels
    y_pred  : hard predictions
    y_proba : positive-class probabilities

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, pr_auc
    """

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_proba),
        "pr_auc":    average_precision_score(y_true, y_proba),
    }

def plot_roc_curve(y_true: np.ndarray, 
                   y_proba: np.ndarray, 
                   model_name: str
                   ) -> plt.Figure:
    """
    Generate a ROC curve figure for a given model.

    Returns a matplotlib Figure (caller is responsible for plt.close).
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC: {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {model_name}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

def save_artifact(obj: Any, path: str) -> None:
    """
    Serialize any Python object to disk using skops

    Creates parent directories automatically.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # skops handles the file opening/closing internally
    sio.dump(obj, path)

def load_artifact(path: str) -> Any:
    """
    Load a skops-serialized object from disk securely.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No artifact found at {path}")
    return sio.load(path, trusted=True)
    

def build_param_grids(config: dict, y_train: np.ndarray) -> Dict[str, Dict]:
    """
    Build hyperparameter grids for tree-based models by reading from
    config.yaml. Dynamically injects `scale_pos_weight` for XGBoost
    based on the class imbalance ratio in y_train.

    Parameters
    ----------
    config  : full parsed config.yaml dict
    y_train : training target array (used to compute class ratio)

    Returns
    -------
    dict keyed by model name matching keys used in ModelTrainer
    """
    hp = config["hyperparameters"]
    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    grids = {
        "Random Forest": dict(hp["random_forest"]),
        "XGBoost":       dict(hp["xgboost"]),
        "LightGBM":      dict(hp["lightgbm"]),
    }

    # Inject dynamic value not stored in config
    grids["XGBoost"]["scale_pos_weight"] = [spw]

    return grids
