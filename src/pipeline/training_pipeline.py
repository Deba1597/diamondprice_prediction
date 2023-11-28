import sys
import os 
import pandas as pd

from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging

class TrainingPipeline:
    
    def start_data_ingestion(self):
        logging.info('Data ingestion started')
        
        try:
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            return train_data_path, test_data_path
        except Exception as e:
            logging.info('Exception arise at data ingestion in TrainingPipeline Class ')
            raise CustomException(e, sys)


    def start_data_transformation(self, train_data_path, test_data_path):
        logging.info('Data transformation start.')

        try:
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path=train_data_path,
                                                                                      test_path=test_data_path)
            
            return train_arr, test_arr
        
        except Exception as e:
            logging.info(' Exception arise at start_data_transformation in TrainingPipieline class')
            raise CustomException(e, sys)

    def start_model_training(self, train_arr, test_arr):
        logging.info('model training started')
        try:
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_training(train_arr=train_arr, test_arr=test_arr)
        
        except Exception as e:
            logging.info('Exception arise at model training in TrainingPieline class')
            raise CustomException(e, sys)
        

    def run_pipeline(self):
        logging.info('training pieline started')
        try:
            train_data_path, test_data_path = self.start_data_ingestion()
            train_arr, test_arr = self.start_data_transformation(train_data_path, test_data_path)
            self.start_model_training(train_arr,test_arr)

        except Exception as e:
            logging.info('Exception arise at training pipeline')
            raise CustomException(e, sys)
        

# c = TrainingPipeline()
# c.run_pipeline()