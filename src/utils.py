import os
import sys
import pickle 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def eval_score(model, x_test, y_test):
    try:
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return accuracy, precision, recall, f1

    except Exception as e:
        raise CustomException(e, sys)
  


def model_training(models, x_train, y_train, x_test, y_test, params):
    try:
        report = {}
    
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            preds = model.predict(x_test)
            accuracy = accuracy_score(y_test, preds)

            report[list(models.keys())[i]] = accuracy

        return report

    except Exception as e:
        raise CustomException(e, sys)





