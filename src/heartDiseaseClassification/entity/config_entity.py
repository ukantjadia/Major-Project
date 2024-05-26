from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    ptbxl_csv_path: Path
    scp_csv_path: Path
    superclass_csv_data: Path
    freq_rate: int
    ecg_data_path: str
    aug_data_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    model_save_path: Path
    input_size0: int
    input_size1: int
    output_size: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    # all_params: dict
