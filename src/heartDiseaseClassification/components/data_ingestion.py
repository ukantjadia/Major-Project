import requests
import urllib.request as request
import zipfile
import os
from heartDiseaseClassification import logger
from heartDiseaseClassification.utils.common import get_size
from pathlib import Path
from heartDiseaseClassification.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_progress_hook(self, count, blockSize, totalSize):
        old_percent = 0
        percent = int(count * blockSize * 100 / totalSize)
        if percent > old_percent:
            old_percent = percent
            os.system('cls')
            print(percent, '%')
        if percent == 100:
            os.system('cls')
            print('done!')
            
    def download_file(self):
        '''
        Fetch data from the url
        '''
        logger.info(f"downloading the data from url {self.config.source_URL}")
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file,
                reporthook=self.download_progress_hook,

            )
            logger.info(f"{filename}, download! with the following info {headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        logger.info(f"extracting the data from {self.config.local_data_file}")
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        data_path = Path('.'.join(self.config.local_data_file.split('.')[:-1]))
        dir_size = get_size(data_path)
        logger.info(f"Size of current extracted dir is {dir_size}")
        extract_again =input("Want to re-do the extraction ??  1 - yes, other - No ")
        if extract_again == '1':
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Extraction of the data complete!! at location {unzip_path}")
        else: 
            logger.info(f"Continue without Extraction.")