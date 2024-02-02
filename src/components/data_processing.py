import os
import sys
import numpy as np
import pandas as pd


from imblearn.combine import SMOTETomek
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


@dataclass
class DataProcessingConfig():
    preprocessing_obj_path = os.path.join('data/artifacts', 'preprocessor.pkl')


class DataProcessing():
    def __init__(self):
        self.data_processing_config = DataProcessingConfig()

    def get_data_processor(self, train_data):
        try:
            categorical_columns = ['Geography', 'Gender']
            numerical_columns = [
                col for col in train_data.columns if col not in categorical_columns + ['Exited']]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ('scalar', StandardScaler())

            ])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f"Numerical Columns:{numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]

            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_processing(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Reading the train and test file")

            # nan_values = train_df[train_df.isna()]
            # if np.any(nan_values):
            #     print("NaN values found in the initial train dataset")
            # else:
            #     print("No NaN values in the initial train dataset.")
            #     print(print(train_df.dtypes))

            preprocessing_obj = self.get_data_processor(train_df)

            target_col = 'Exited'
            train_input_features = train_df.drop(columns=[target_col], axis=1)
            train_target_features = train_df[target_col]
            test_input_features = test_df.drop(columns=[target_col], axis=1)
            test_target_features = test_df[target_col]

            # nan_values1 = train_input_features[train_input_features.isna()]
            # if np.any(nan_values1):
            #     print("NaN values found in the input features.")
            # else:
            #     print("No NaN values found after droping the target feature.")
            #     print(train_input_features.head(5))

            logging.info(
                "Applying Preprocessing on training and test dataframe")
            train_input_feature_arr = preprocessing_obj.fit_transform(
                train_input_features)
            test_input_feature_arr = preprocessing_obj.transform(
                test_input_features)

            # print(f"train_input_feature_arr: {train_input_feature_arr}")
            # print(f"train_target_features: {train_target_features}")

            # nan_values2 = np.isnan(train_input_feature_arr)
            # if np.any(nan_values2):
            #     print(f"NaN values found in the input feature after preprocessing.")
            #     print("total nan values: ", len(nan_values2))
            #     print("shape of the preprocessed features: ", train_input_feature_arr.shape)
            #     print("shape befor preprocessing the features: ", train_input_features.shape)
            # else:
            #     print("No NaN values in the input features.")

            logging.info("Balancing the training dataset.")
            smk = SMOTETomek(random_state=42)
            train_input_feature_arr, train_target_features = smk.fit_resample(
                train_input_feature_arr, train_target_features)

            save_object(

                file_path=self.data_processing_config.preprocessing_obj_path,
                obj=preprocessing_obj
            )

            return (
                train_input_feature_arr,
                test_input_feature_arr,
                train_target_features,
                test_target_features,
                self.data_processing_config.preprocessing_obj_path
            )

        except Exception as e:
            raise CustomException(e, sys)
