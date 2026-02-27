import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import yaml
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, make_scorer
)
import xgboost as xgb
import lightgbm as lgb

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    compute_metrics,
    plot_roc_curve,
    save_artifact,
    load_artifact,
    build_param_grids,
)

logging.info('Load config once at module level')
with open("config.yaml", "r") as _f:
    _CONFIG = yaml.safe_load(_f)

_MT_CFG = _CONFIG["model_trainer"]

@dataclass
class ModelTrainerConfig:
    mlflow_experiment_name: str  = _MT_CFG["mlflow_experiment_name"]
    trained_model_file_path: str = _MT_CFG["trained_model_file_path"]
    mlflow_tracking_uri: str     = _MT_CFG["mlflow_tracking_uri"]
    n_iter_search: int           = _MT_CFG["n_iter_search"]
    cv_folds: int                = _MT_CFG["cv_folds"]
    primary_metric: str          = _MT_CFG["primary_metric"]
    random_state: int            = _MT_CFG["random_state"]


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.cfg = config
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.cfg.mlflow_experiment_name)

    
    def _train_baseline(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ) -> Tuple[LogisticRegression, Dict[str, float]]:
        
        """Train a baseline Logistic Regression and log it to MLflow."""
        logging.info("Training baseline Logistic Regression ...")

        model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=self.cfg.random_state,
        )
        model.fit(X_train, y_train)

        y_val_pred  = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        metrics     = compute_metrics(y_val, y_val_pred, y_val_proba) 

        logging.info(f"[Baseline LR] val metrics: {metrics}")
        logging.info("\n" + classification_report(y_val, y_val_pred))

        with mlflow.start_run(run_name="Baseline_LogisticRegression", nested=True):
            mlflow.log_params({
                "model":        "LogisticRegression",
                "solver":       "liblinear",
                "class_weight": "balanced",
                "split":        "validation",
            })
            mlflow.log_metrics(metrics)

            roc_fig = plot_roc_curve(y_val, y_val_proba, "Logistic Regression") 
            mlflow.log_figure(roc_fig, "roc_curve.png")
            plt.close(roc_fig)

            mlflow.sklearn.log_model(model, name="base_model")

        return model, metrics



    def _tune_models(
            self,
            models:      Dict[str, Any],
            param_grids: Dict[str, Dict],
            X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray,
        ) -> pd.DataFrame:
            
            """Run RandomizedSearchCV for each tree-based model and log to MLflow."""

            logging.info("Starting hyperparameter search for tree-based models ...")

            fraud_recall_scorer = make_scorer(recall_score, pos_label=1)
            tuning_report = []

            for model_name, model_obj in models.items():
                logging.info(f"Tuning: {model_name}")

                rs = RandomizedSearchCV(
                    estimator=model_obj,
                    param_distributions=param_grids[model_name],
                    n_iter=self.cfg.n_iter_search,
                    scoring=fraud_recall_scorer,
                    cv=self.cfg.cv_folds,
                    n_jobs=-1,
                    random_state=self.cfg.random_state,
                    verbose=1,
                )
                rs.fit(X_train, y_train)

                best_model  = rs.best_estimator_
                y_val_pred  = best_model.predict(X_val)
                y_val_proba = best_model.predict_proba(X_val)[:, 1]
                metrics     = compute_metrics(y_val, y_val_pred, y_val_proba)   # <- utils.py

                logging.info(f"[{model_name}] best params : {rs.best_params_}")
                logging.info(f"[{model_name}] val metrics : {metrics}")

                with mlflow.start_run(run_name=f"Tuning_{model_name.replace(' ', '_')}", nested=True):
                    mlflow.log_params(rs.best_params_)
                    mlflow.log_params({"model": model_name, "split": "validation"})
                    mlflow.log_metrics(metrics)

                    roc_fig = plot_roc_curve(y_val, y_val_proba, model_name)    
                    mlflow.log_figure(roc_fig, "roc_curve.png")
                    plt.close(roc_fig)

                    # Framework-specific loggers for richer MLflow artefacts
                    if isinstance(best_model, xgb.XGBClassifier):
                        mlflow.xgboost.log_model(best_model, name="model")
                    elif isinstance(best_model, lgb.LGBMClassifier):
                        mlflow.lightgbm.log_model(best_model, name="model")
                    else:
                        mlflow.sklearn.log_model(best_model, name="model")

                tuning_report.append({
                    "model":       model_name,
                    "best_model":  best_model,
                    "best_params": rs.best_params_,
                    **metrics,
                })

            return pd.DataFrame(tuning_report)

    def _evaluate_on_test(
            self,
            tuning_df: pd.DataFrame,
            X_test: np.ndarray,
            y_test: np.ndarray,
        ) -> pd.DataFrame:
            
            """Re-evaluate every tuned model on the test set and log to MLflow."""
            logging.info("Final evaluation on test set ...")
            test_results = []

            for _, row in tuning_df.iterrows():
                model_name = row["model"]
                best_model = row["best_model"]

                y_pred  = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1]
                metrics = compute_metrics(y_test, y_pred, y_proba)         

                logging.info(f"\n=== {model_name} - TEST ===")
                logging.info("\n" + classification_report(y_test, y_pred))
                logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

                with mlflow.start_run(run_name=f"Test_{model_name.replace(' ', '_')}", nested=True):
                    mlflow.log_params({"model": model_name, "split": "test"})
                    mlflow.log_metrics(metrics)

                    roc_fig = plot_roc_curve(y_test, y_proba, model_name)   
                    mlflow.log_figure(roc_fig, "roc_curve_test.png")
                    plt.close(roc_fig)

                test_results.append({
                    "model":      model_name,
                    "best_model": best_model,
                    **metrics,
                })

            return pd.DataFrame(test_results)

    def _select_and_save_best(
            self,
            lr_model:   LogisticRegression,
            lr_metrics: Dict[str, float],
            test_df:    pd.DataFrame,
        ) -> Tuple[Any, str, Dict[str, float]]:
            
            """
            Build a leaderboard across all models, pick the winner by
            `primary_metric` (set in config.yaml), and persist the model.
            """

            metric = self.cfg.primary_metric

            candidates = [
                {"model_name": "Logistic Regression", "model_obj": lr_model, **lr_metrics}
            ]
            for _, row in test_df.iterrows():
                candidates.append({
                    "model_name": row["model"],
                    "model_obj":  row["best_model"],
                    **{k: row[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]},
                })

            leaderboard = pd.DataFrame(candidates).sort_values(metric, ascending=False)
            logging.info(
                f"\nLeaderboard (sorted by {metric}):\n"
                + leaderboard[["model_name", metric, "f1", "roc_auc"]].to_string(index=False)
            )

            best_row     = leaderboard.iloc[0]
            best_name    = best_row["model_name"]
            best_model   = best_row["model_obj"]
            best_metrics = {
                k: best_row[k]
                for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
            }

            logging.info(f"Best model: {best_name} | {metric} = {best_metrics[metric]:.4f}")

            save_artifact(best_model, self.cfg.trained_model_file_path)     
            logging.info(f"Best model saved -> {self.cfg.trained_model_file_path}")

            return best_model, best_name, best_metrics


    def initiate_model_trainer(
            self,
            train_log_arr:  np.ndarray,
            val_log_arr:    np.ndarray,
            test_log_arr:   np.ndarray,
            train_tree_arr: np.ndarray,
            val_tree_arr:   np.ndarray,
            test_tree_arr:  np.ndarray,
        ) -> Tuple[Any, str, Dict[str, float]]:
            
            """
            Full pipeline: baseline -> tuning -> test evaluation -> best-model selection.

            Each array must have features in all columns except the last,
            which should be the binary target label.

            Returns
            -------
            best_model   : fitted estimator
            best_name    : winning model name
            best_metrics : dict of test-set metrics for the winner
            """
            try:
                # Unpack arrays
                logging.info("Unpacking train / val / test arrays ...")

                X_train_scal, y_train_scal = train_log_arr[:, :-1],  train_log_arr[:, -1]
                X_val_scal,   y_val_scal   = val_log_arr[:,   :-1],  val_log_arr[:,   -1]
                X_test_scal,  y_test_scal  = test_log_arr[:,  :-1],  test_log_arr[:,  -1]

                X_train, y_train = train_tree_arr[:, :-1], train_tree_arr[:, -1]
                X_val,   y_val   = val_tree_arr[:,   :-1], val_tree_arr[:,   -1]
                X_test,  y_test  = test_tree_arr[:,  :-1], test_tree_arr[:,  -1]

                # Build param grids from config.yaml; utils injects dynamic scale_pos_weight
                param_grids = build_param_grids(_CONFIG, y_train)               # <- utils.py

                tree_models = {
                    "Random Forest": RandomForestClassifier(),
                    "XGBoost":       xgb.XGBClassifier(),
                    "LightGBM":      lgb.LGBMClassifier(),
                }

                # Parent MLflow run wraps all nested runs
                with mlflow.start_run(run_name="fraud_detection_experiment"):

                    # 1. Baseline (scaled data for Logistic Regression)
                    lr_model, _ = self._train_baseline(
                        X_train_scal, y_train_scal,
                        X_val_scal,   y_val_scal,
                    )

                    # 2. Hyperparameter tuning (unscaled data for tree models)
                    tuning_df = self._tune_models(
                        tree_models, param_grids,
                        X_train, y_train,
                        X_val,   y_val,
                    )

                    # 3. Test-set evaluation for all tree models
                    test_df = self._evaluate_on_test(tuning_df, X_test, y_test)

                    # Evaluate baseline on test set for a fair comparison
                    lr_test_pred    = lr_model.predict(X_test_scal)
                    lr_test_proba   = lr_model.predict_proba(X_test_scal)[:, 1]
                    lr_test_metrics = compute_metrics(                      
                        y_test_scal, lr_test_pred, lr_test_proba
                    )

                    # 4. Pick winner and persist to artifacts/
                    best_model, best_name, best_metrics = self._select_and_save_best(
                        lr_model, lr_test_metrics, test_df
                    )

                    # Surface winner metrics in the parent MLflow run
                    mlflow.log_param("best_model", best_name)
                    mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})

                return best_model, best_name, best_metrics

            except Exception as e:
                raise CustomException(e, sys)