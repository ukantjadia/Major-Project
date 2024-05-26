from heartDiseaseClassification import logger
from heartDiseaseClassification.components.model_trainign import ModelTrainer
from heartDiseaseClassification.config.configuration import ConfigurationManager

STAGE_NAME = "Model Traning Stage"

class ModelTrainerTrainingipeline:
    def __init__(self):
        pass 
    
    def main(self):
        config = ConfigurationManager()
        model_traning_config = config.get_model_trainer_config()
        model_traning_config = ModelTrainer(config=model_traning_config)
        model_traning_config.train_model()
        
        
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainerTrainingipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<  \n\n x=======x")
    except Exception as e:
        logger.exception(e)
        raise e  