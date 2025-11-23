import sys
from pathlib import Path
import os
from logging import Logger
sys.path.append(str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI

from document_ocr import DocumentOcr
from config import Config
from custom_logger import CustomLogger

def setup_logging() -> Logger:
    """Configure logging for the api"""
    log_dir = os.path.dirname(Config.API_LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = CustomLogger(
        logger_name="middleware_logger",
        logger_log_level=Config.CLI_LOG_LEVEL,
        file_handler_log_level=Config.FILE_LOG_LEVEL,
        log_file_name=Config.API_LOG_FILE
    ).create_logger()

    return logger

logger = setup_logging()
logger.info("Starting api...")

app = FastAPI(title="OcrDocumentsApi")
document_ocr = DocumentOcr(
    ocr_model_path=f"{Config.OCR_MODELS_FOLDER_PATH}{Config.OCR_MODEL_NAME}",
    yolo_model_path=f"{Config.YOLO_MODELS_FOLDER_PATH}{Config.YOLO_MODEL_NAME}",
    yolo_classes_path=Config.YOLO_CLASSES_FILE,
    yolo_device=Config.DEVICE
)

from ocr_api import routes
logger.info("Api started")