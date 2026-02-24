import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "train.csv")
    val_data_path: str=os.path.join('artifacts', "val.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_raw_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r"notebook\data\raw_data.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split has been initiated")
        
            return self.ingestion_config.raw_data_path
        
        except Exception as e:
            raise CustomException (e,sys)
        
    
    def initiate_train_val_test_data_ingestion(self):
        logging.info("Train, val, test data ingestion inititated")
        
        try:
            obj = DataIngestion()
            raw_data = pd.read_csv(r"artifacts\data.csv")
            logging.info(f'data contains {len(raw_data)} rows and {len(raw_data.columns)} columns')

            logging.info(f'Remove the duplicated rows {raw_data.duplicated().sum()}')
            df_clean = raw_data.drop_duplicates()
            logging.info(f'data contains {len(df_clean)} rows and {len(df_clean.columns)} columns after removing the duplicated rows')

            y = df_clean['Class']
            train_set, temp_set = train_test_split( df_clean, test_size=0.4, stratify=y, random_state=42)

            y_temp = temp_set['Class']
            val_set, test_set = train_test_split(temp_set, test_size=0.5, stratify=y_temp, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f'Number of rows in train dataset {train_set.shape[0]}, Number of columns in train feature dataset {train_set.shape[1]} and the percentage of Fraud {len(train_set[train_set['Class']==1])/len(train_set)*100:.3f}%')
            logging.info(f'Number of rows in validation dataset {val_set.shape[0]}, Number of columns in validation feature dataset {val_set.shape[1]} and the percentage of Fraud {len(val_set[val_set['Class']==1])/len(val_set)*100:.3f}%')
            logging.info(f'Number of rows in test dataset {test_set.shape[0]}, Number of columns in test feature dataset {test_set.shape[1]} and the percentage of Fraud {len(test_set[test_set['Class']==1])/len(test_set)*100:.3f}%')

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException (e,sys)
        


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_train_val_test_data_ingestion()
