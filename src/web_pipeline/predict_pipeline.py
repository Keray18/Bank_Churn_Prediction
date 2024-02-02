import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def initiate_prediction(self, features):
        try:
            model_path = os.path.join("data/artifacts", "model.pkl")
            preprocessor_path = os.path.join(
                "data/artifacts", "preprocessor.pkl")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            processed_data = preprocessor.transform(features)
            preds = model.predict(processed_data)
            return preds

        except Exception as e:
            raise CustomException
