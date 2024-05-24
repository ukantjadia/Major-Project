from pathlib import Path 
from ensure import ensure_annotations
from box import ConfigBox
from heartDiseaseClassification import logger
from box.exceptions import BoxValueError
from typing import Any

import os 
import json
import yaml
import joblib

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml files: {path_to_yaml} loadded successfully!!")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories:list, verbos=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbos:
            logger.info(f"created director at : {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path) as file:
        content = json.load(file)
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(path: Path,data: Any):
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path) as file:
        content = json.load(file)
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path) / 1024)
    logger.info(f"Size: {size_in_kb} file: {path}")
    return f"~ {size_in_kb} KB"