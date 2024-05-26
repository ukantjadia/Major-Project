from heartDiseaseClassification import logger
from heartDiseaseClassification.components.data_transformation import Datatransformation
from heartDiseaseClassification.config.configuration import ConfigurationManager

STAGE_NAME = "Data Transformation stage"

class DataTransformationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = Datatransformation(config=data_transformation_config)
        # data_transformation.load_ptbxl_csv_file()
        logger.info(f"running train test spilit")
        # data_transformation.train_test_split()
        logger.info(f"Doing the data augmentation")
        data_transformation.applying_SWT()
        
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<  \n\n x=======x")
    except Exception as e:
        logger.exception(e)
        raise e        