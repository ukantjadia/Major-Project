import os 
import sys
import logging 
import colorlog 

formater = colorlog.ColoredFormatter(
    "[%(asctime)s: %(log_color)s%(levelname)s%(reset)s: %(module)s: %(message)s]",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG":"cyan",
        "INFO":"green",
        "WARNING":"yellow",
        "ERROR":"red",
        "CRITICAL":"bold_red",
    }
)

log_dir = "logs"
logging_str = "%(asctime)s: %(levelname)s: %(module)s: %(message)s"
log_filepath = os.path.join(log_dir,"running_log.log")
os.makedirs(log_dir,exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
    ]
)


console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formater)


logger = logging.getLogger("heartDiseaseClassification")
logger.addHandler(console_handler)