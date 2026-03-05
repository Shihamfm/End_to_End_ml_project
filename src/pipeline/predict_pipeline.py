"""
predict_pipeline.py
====================
Fixes applied vs previous version:
  1. PredictPipeline split into __init__ (load once) + predict() (call many times).
     Previously __init__ did everything including disk I/O, making it reload
     the model on every single row prediction call from app.py.
  2. CustomData trailing-comma tuple bug removed from all V1–V28 assignments.
     Previously: self.V1 = V1,  -->  stored (float,) tuple instead of float.
     Now:        self.V1 = V1   -->  stores float as intended.
"""

import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_artifact


# ---------------------------------------------------------------------------
# FIX 1 — Split __init__ (disk I/O) from predict() (inference)
# ---------------------------------------------------------------------------

class PredictPipeline:
    """
    Loads the model and preprocessor once in __init__.
    predict() can then be called many times with zero additional disk I/O.
    """

    def __init__(self):
        try:
            model_path        = os.path.join("artifacts", "best_model.skops")
            preprocessor_path = os.path.join("artifacts", "preprcessor_tree.pkl")

            logging.info("Loading model from %s", model_path)
            self.model = load_artifact(model_path)

            logging.info("Loading preprocessor from %s", preprocessor_path)
            self.preprocessor = load_artifact(preprocessor_path)

            logging.info("PredictPipeline initialised successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Preprocess and predict on a full DataFrame in one vectorised call.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame with columns matching the training feature set.

        Returns
        -------
        np.ndarray of shape (n_rows,) with values 0 (legit) or 1 (fraud).
        """
        try:
            logging.info("Preprocessing %d row(s)", len(features))
            data_preprocessed = self.preprocessor.transform(features)

            logging.info("Running model inference")
            predictions = self.model.predict(data_preprocessed)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


# ---------------------------------------------------------------------------
# FIX 2 — Remove trailing commas from all V-column assignments
# ---------------------------------------------------------------------------

class CustomData:
    """
    Stores a single transaction's raw field values and converts them to a
    one-row DataFrame for the prediction pipeline.

    Bug fixed: trailing commas after assignments (e.g. self.V1 = V1,) were
    silently wrapping every value in a tuple, causing wrong model inputs.
    """

    def __init__(self,
                 Time:   int,
                 V1:     float, V2:  float, V3:  float, V4:  float,
                 V5:     float, V6:  float, V7:  float, V8:  float,
                 V9:     float, V10: float, V11: float, V12: float,
                 V13:    float, V14: float, V15: float, V16: float,
                 V17:    float, V18: float, V19: float, V20: float,
                 V21:    float, V22: float, V23: float, V24: float,
                 V25:    float, V26: float, V27: float, V28: float,
                 Amount: float):

        self.Time   = Time
        # FIX: no trailing commas — these now store float, not (float,) tuple
        self.V1     = V1
        self.V2     = V2
        self.V3     = V3
        self.V4     = V4
        self.V5     = V5
        self.V6     = V6
        self.V7     = V7
        self.V8     = V8
        self.V9     = V9
        self.V10    = V10
        self.V11    = V11
        self.V12    = V12
        self.V13    = V13
        self.V14    = V14
        self.V15    = V15
        self.V16    = V16
        self.V17    = V17
        self.V18    = V18
        self.V19    = V19
        self.V20    = V20
        self.V21    = V21
        self.V22    = V22
        self.V23    = V23
        self.V24    = V24
        self.V25    = V25
        self.V26    = V26
        self.V27    = V27
        self.V28    = V28
        self.Amount = Amount

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "Time":   [self.Time],
                "V1":     [self.V1],  "V2":  [self.V2],  "V3":  [self.V3],  "V4":  [self.V4],
                "V5":     [self.V5],  "V6":  [self.V6],  "V7":  [self.V7],  "V8":  [self.V8],
                "V9":     [self.V9],  "V10": [self.V10], "V11": [self.V11], "V12": [self.V12],
                "V13":    [self.V13], "V14": [self.V14], "V15": [self.V15], "V16": [self.V16],
                "V17":    [self.V17], "V18": [self.V18], "V19": [self.V19], "V20": [self.V20],
                "V21":    [self.V21], "V22": [self.V22], "V23": [self.V23], "V24": [self.V24],
                "V25":    [self.V25], "V26": [self.V26], "V27": [self.V27], "V28": [self.V28],
                "Amount": [self.Amount],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)