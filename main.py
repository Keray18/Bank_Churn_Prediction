import sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.components.data_processing import DataProcessingConfig, DataProcessing
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

if __name__=='__main__':
    logging.info("Execution has started...")
    try:
        data_ingestion=DataIngestion()
        train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()

        data_processing = DataProcessing()
        x_train, x_test, y_train, y_test, _ = data_processing.initiate_data_processing(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_training(x_train, x_test, y_train, y_test))

    except Exception as e:
        raise CustomException(e,sys)