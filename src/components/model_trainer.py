import os
import sys   
import numpy as np
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'newton-cg'],
                    # 'max_iter': [100, 200, 300]
                },
                'XGBoost Classifier': {
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    # 'subsample': [0.8, 0.9, 1.0],
                    # 'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'Random Forest Classifier': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4],

                }
            }

            report: dict = model_training(models, x_train, y_train, x_test, y_test, params)

            # Getting best model name.
            best_score = max(sorted(report.values()))
            print(f"best_score: {best_score}")
            
            best_model_name = list(report.keys())[
                list(report.values()).index(best_score)
            ]
            print(f"best_model_name: {best_model_name}")

            # Getting the best model.
            best_model = models[best_model_name] 
            logging.info(f"Best model found")
            print(f"Best model is: {best_model_name} with accuracy: {best_score}")

            model_names = list(params.keys())
            actual_model=""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]

            mlflow.set_registry_uri("https://dagshub.com/Keray18/Bank_Churn_Prediction.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            dagshub.init(repo_owner='Keray18', repo_name='Bank_Churn_Prediction', mlflow=True)
            with mlflow.start_run():
                accuracy, precision, recall, f1 = eval_score(best_model, x_test, y_test)

                mlflow.log_params(best_params)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                
                else:
                    mlflow.sklearn.log_model(best_model, "model")

                if best_score<0.6:
                    raise CustomException("No best model found")
                logging.info(f"Best found model on both training and testing dataset")

                # print(f"evalScore of the best model: {evalScore}")
            save_object (
                file_path = self.model_trainer_config.model_file_path,
                obj = best_model
            )

            preds=best_model.predict(x_test)

            accuracy = accuracy_score(y_test, preds)
            return accuracy




        except Exception as e:
            raise CustomException(e,sys)