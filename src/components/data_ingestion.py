import os  
import sys 
import pandas as pd
import requests

from pathlib import Path
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig():
    train_data_file_path: str = os.path.join('data/artifacts', 'train.csv')
    test_data_file_path: str = os.path.join('data/artifacts', 'test.csv')


class DataIngestion():
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion...")
            df = pd.read_csv("data\Churn_Modelling.csv")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_file_path), exist_ok=True)
            
            # Dropping non-esential features.
            low_correlated_features = ["RowNumber", "CustomerId", "Surname", "EstimatedSalary", "HasCrCard", "Tenure"]
            df = df.drop(low_correlated_features, axis=1)

            logging.info("Splitting dataset into training and test data.")
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            train_data.to_csv(self.data_ingestion_config.train_data_file_path, index=False, header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_file_path, index=False, header=True)
            logging.info("Data Ingestion is Successful! Train and test data are saved.")

            return (
                self.data_ingestion_config.train_data_file_path,
                self.data_ingestion_config.test_data_file_path
            )


        except Exception as e:
            raise CustomException(e,sys) 