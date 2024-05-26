import joblib
from pathlib import Path

from tensorflow.keras.models import load_model

class PredictionPipeline:
    def __init__(self):
        self.model = load_model(f"artifacts/model_training/saved_model/ECG-Classifier.h5")
        # self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))
        
    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction