from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from heartDiseaseClassification import logger
from tensorflow.keras.models import load_model
import numpy as np 
from heartDiseaseClassification.entity.config_entity import ModelEvaluationConfig
from pathlib import Path 
from heartDiseaseClassification.utils.common import save_json

from tensorflow.keras.utils import to_categorical

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config = config
        
    def eval_matrix(self,y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        return accuracy,precision,recall,f1 
    
    def evaluation(self):
        
        # model = joblib.load(f"{self.config.model_path}.pkl")
        model = load_model(f"{self.config.model_path}.h5")
        test_x = np.load(f"{self.config.test_data_path}/X_test.npy")
        test_y = np.load(f"{self.config.test_data_path}/y_test.npy")
        test_y = to_categorical(test_y,num_classes=5)
        
        predicted_qualities = model.predict(test_x)
        
        ( accuracy,precision,recall,f1) = self.eval_matrix(test_y,predicted_qualities)
        logger.info(f"+++accuracy {accuracy}, precision {precision}, recall {recall} f1 {f1} +++")
            
        scores = {"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1}
        save_json(path=Path(self.config.metric_file_name),data=scores)