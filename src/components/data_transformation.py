import os
import sys
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.components.data_ingestion import DataIngestion


class DataTrasnformationConfig:
    preprocessor_obj_file_path_log = os.path.join('artifacts', "preprcessor_log.pkl")
    preprocessor_obj_file_path_tree = os.path.join('artifacts', "preprcessor_tree.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTrasnformationConfig()
    
    def get_data_transformer_object(self):
        '''
        The function is responsible for data transformation
        '''
        try:
            numerical_columns = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']
            num_pipeline_log =  Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            num_pipeline_tree =  Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]
            )
            logging.info("Preprocessing the columns for base model (logistic regression)")
            preprocessor_log = ColumnTransformer(
                [
                    ("preprocess_log_model", num_pipeline_log, numerical_columns)
                ]
            )

            logging.info("Preprocessing the columns for tree base models")
            preprocessor_tree = ColumnTransformer(
                [
                    ("preprocess_tree_model", num_pipeline_tree, numerical_columns)
                ]
            )

            return preprocessor_log, preprocessor_tree

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, val_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train, validation & test data are completed")

            logging.info("Obtaining preprocessing object")
            preprcessing_obj_log, preprcessing_obj_tree = self.get_data_transformer_object()
            
            target_column = "Class"
            input_features = train_df.drop(['Class','Time'],axis=1)
            input_feature_columns = input_features.columns.tolist()

            logging.info("Creating training, validation and testing feature dataset")
            X_train = train_df[input_feature_columns]
            X_val = val_df[input_feature_columns]
            X_test = test_df[input_feature_columns]

            logging.info("Applying preprocessing object on training, validation and testing dataframe")
            
            X_train_log_arr = preprcessing_obj_log.fit_transform(X_train)
            X_val_log_arr = preprcessing_obj_log.transform(X_val)
            X_test_log_arr = preprcessing_obj_log.transform(X_test)

            X_train_tree_arr = preprcessing_obj_tree.fit_transform(X_train)
            X_val_tree_arr = preprcessing_obj_tree.transform(X_val)
            X_test_tree_arr = preprcessing_obj_tree.transform(X_test)

            logging.info("Applying zipper to concatenate feature columns with target columns on training, validation and testing dataframe")

            train_log_arr = np.c_[X_train_log_arr, np.array(train_df[target_column])]
            val_log_arr = np.c_[X_val_log_arr, np.array(val_df[target_column])]
            test_log_arr = np.c_[X_test_log_arr, np.array(test_df[target_column])]

            train_tree_arr = np.c_[X_train_tree_arr, np.array(train_df[target_column])]
            val_tree_arr = np.c_[X_val_tree_arr, np.array(val_df[target_column])]
            test_tree_arr = np.c_[X_test_tree_arr, np.array(test_df[target_column])]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path_log,
                obj = preprcessing_obj_log
            )

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path_tree,
                obj = preprcessing_obj_tree
            )

            return train_log_arr, val_log_arr, test_log_arr, train_tree_arr, val_tree_arr, test_tree_arr, self.data_transformation_config.preprocessor_obj_file_path_log, self.data_transformation_config.preprocessor_obj_file_path_tree

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj_data_ingestion = DataIngestion()
    train_data_path, val_data_path, test_data_path = obj_data_ingestion.initiate_train_val_test_data_ingestion()

    obj_data_transformation = DataTransformation()
    data_transformation = obj_data_transformation.initiate_data_transformation(train_data_path, val_data_path, test_data_path)


            