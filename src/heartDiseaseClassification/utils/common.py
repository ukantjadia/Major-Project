import os
from box import exceptions 
# from box.exceptions import BoxValueError
import yaml
from heartDiseaseClassification import logger
import json 
import joblib 
from ensure import ensure_annotations
from box import ConfigBox 
from pathlib import Path  
from typing import Any 

@ensure_annotations
def get_size(path:Path) -> str:
    """get the size of 

    Args:
        path (Path): path of file

    Returns:
        Any: size in kB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    logger.info(f"Size: {size_in_kb} file: {path}")
    return f"~ {size_in_kb} KB"

@ensure_annotations
def load_bin(path:Path) -> Any:
    """load the binary file

    Args:
        path (Path): path of binary file

    Returns:
        Any: data of binary file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data 

@ensure_annotations
def save_bin(data:Any, path:Path):
    """save binary file

    Args:
        data (Any): data to save as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json file data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)
        
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_json(path: Path, data: dict):
    """save output to json file

    Args:
        path (Path): path of json file
        data (dict): output data
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """_summary_

    Args:
        path_to_directories (list): path of directory
        verbose (bool, optional): creating logs
    """
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")
        
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Args:
        path_to_yaml (Path): yaml path

    Raises:
        ValueError: if yaml is empty
        e: error 'e' 

    Returns:
        ConfigBox: yaml content 
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except exceptions.BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e 
    
    
    
