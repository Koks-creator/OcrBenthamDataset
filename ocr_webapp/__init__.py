from typing import Tuple
import sys
import os
import logging
from pathlib import Path
from flask import Flask
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

def setup_logging(app: Flask) -> None:
    """Configure logging for the application"""
    log_dir = os.path.dirname(Config.WEB_APP_LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    file_handler = logging.FileHandler(Config.WEB_APP_LOG_FILE)
    file_handler.setLevel(Config.FILE_LOG_LEVEL)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - Line: %(lineno)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    
    app.logger.addHandler(file_handler)
    app.logger.setLevel(Config.FILE_LOG_LEVEL)
    app.logger.propagate = False  # Prevent duplicate logs


def create_app() -> Tuple[Flask]:
    """Initialize and configure the Flask application"""
    app = Flask(
        "OcrWebApp",
        template_folder=TEMPLATE_DIR,
        static_folder=STATIC_DIR
    )
    
    setup_logging(app)
    app.logger.info("Starting web app")
    
    app.secret_key = os.getenv("SECRET_KEY")
    app.config["TESTING"] = Config.WEB_APP_TESTING
    
    app.logger.info("Starting predictor")
    app.logger.info("OCR predictor started successfully")
    
    return app

app = create_app()
from ocr_webapp import routes