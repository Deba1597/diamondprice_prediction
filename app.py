import os
import sys

from flask import Flask, render_template, jsonify, request
from src.exception import CustomException
from src.logger import logging

from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome to my appliction'

@app.route('/train')
def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        
        return 'Training Complete'
    except Exception as e:
        raise CustomException(e, sys)

@app.route('/predict',methods=['POST','GET'])
def predict_datapoint():
    
    try:
        if request.method =='POST':
            data = CustomData(
                carat=float(request.form.get('carat')),
                depth = float(request.form.get('depth')),
                table = float(request.form.get('table')),
                x = float(request.form.get('x')),
                y = float(request.form.get('y')),
                z = float(request.form.get('z')),
                cut = request.form.get('cut'),
                color= request.form.get('color'),
                clarity = request.form.get('clarity')
            )
            final_data = data.get_data_as_dataframe()

            prediction_pipeline = PredictionPipeline()
            predicted_price = prediction_pipeline.predict(final_data)

            result = predicted_price

            return render_template('prediction.html', final_result=result)

    
        else:
            return render_template('index.html')
    except Exception as e:
            raise CustomException(e, sys)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)