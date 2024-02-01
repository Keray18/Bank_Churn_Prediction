import os
import sys   
import numpy as np

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from src.utils import save_object, model_training, eval_score
from src.logger import logging  
from src.exception import CustomException
from dataclasses import dataclass 


@dataclass 
class ModelTrainerConfig():
    model_file_path = os.path.join("data/artifact", "model.pkl")


class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, x_train, x_test, y_train, y_test):
        try:
            logging.info("Initiating model training...")

            models = {
                'Logistic Regression': LogisticRegression(),
                'XGBoost Classifier': XGBClassifier(),
                'Random Forest Classifier': RandomForestClassifier()
            }

            params = {
                'Logistic Regression': {
                    'penalty': ['l2', None],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100]
                },
                'XGBoost Classifier': {
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 9]
                },
                'Random Forest Classifier': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30]
                }
            }

            report: dict = model_training(models, x_train, y_train, x_test, y_test, params)

            # Getting best model name.
            best_score = max(sorted(report.values()))
            print(f"best_score: {best_score}")
            
            best_model_name = ""
            for model_name, acc in report.items():
                if acc == best_score:
                    best_model_name = best_model_name + model_name

            
            print(f"best_model_name: {best_model_name}")

            # Getting the best model.
            best_model = models[best_model_name] 
            
            
            logging.info(f"Best model found")
            print(f"Best model is: {best_model_name} with accuracy: {best_score}")

            save_object (
                file_path = self.model_trainer_config.model_file_path,
                obj = best_model
            )
            # evalScore = eval_score(best_model, x_test, y_test)
            # print(f"evalScore of the best model: {evalScore}")





        except Exception as e:
            raise CustomException(e,sys)