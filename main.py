from heartDiseaseClassification import logger

from heartDiseaseClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

# from heartDiseaseClassification.pipeline.stage_02_data_validation import DataValidationTrainingPipeline

# from heartDiseaseClassification.pipeline.stage_03_data_transformation import DataTransformationPipeline

# from heartDiseaseClassification.pipeline.stage_04_model_training import ModelTrainerTrainingipeline

# from heartDiseaseClassification.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline

logger.info(f"welcome to the Heart Classification Project ")


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<<  \n\n x=======x")
except Exception as e:
    logger.exception(e)
    raise e
