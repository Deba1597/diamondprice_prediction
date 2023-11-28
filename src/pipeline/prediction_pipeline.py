import os
import sys
import pickle 
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import load_object

from flask import request
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    model_file_path:str = os.path.join('artifacts', 'model.pkl')
    preprocessor_file_path : str = os.path.join('artifacts', 'preprocessor.pkl')

class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity
    
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'carat':[self.carat],
                    'depth':[self.depth],
                    'table':[self.table],
                    'x':[self.x],
                    'y':[self.y],
                    'z':[self.z],
                    'cut':[self.cut],
                    'color':[self.color],
                    'clarity':[self.clarity]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise CustomException(e, sys)


class PredictionPipeline:
    def __init__(self):
        
        self.prediction_pipeline_config = PredictionPipelineConfig()

        self.model = load_object(self.prediction_pipeline_config.model_file_path)
        self.preprocessor = load_object(self.prediction_pipeline_config.preprocessor_file_path)

    def predict(self, feature):
        try:
            scaled_data = self.preprocessor.transform(feature)
            pred = self.model.predict(scaled_data)

            return pred
        except Exception as e:
            raise CustomException(e, sys)


      

            
        
# abc = PredictionPipeline(request)
# abc.perform_prediction()