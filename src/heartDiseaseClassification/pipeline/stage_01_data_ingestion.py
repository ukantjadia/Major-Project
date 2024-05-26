from heartDiseaseClassification.config.configuration import ConfigurationManager
from heartDiseaseClassification.components.data_ingestion import DataIngestion
from heartDiseaseClassification import logger


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        logger.debug(f"ingestion_config complete")
        data_ingestion = DataIngestion(config=data_ingestion_config)
        logger.debug(f"data_ingestion create, before dowload file")
        data_ingestion.download_file()
        logger.debug(f"download file completed")
        data_ingestion.extract_zip_file()
        logger.debug(f"zip file extracted")
        
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<  \n\n x=======x")
    except Exception  as e:
        logger.exception(e)
        raise e
    