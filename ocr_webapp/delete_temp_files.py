import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from glob import glob
from time import time
import os

from config import Config
from custom_logger import CustomLogger
from custom_decorators import timeit

logger = CustomLogger(logger_log_level=Config.CLI_LOG_LEVEL,
                      file_handler_log_level=Config.FILE_LOG_LEVEL,
                      log_file_name=fr"{Config.ROOT_PATH}/logs/delete_temp_files.log"
                      ).create_logger()

@timeit(logger=logger)
def sztarte() -> None:
    all_files = glob(f"{Config.WEB_APP_TEMP_UPLOADS_FOLDER}/*.*")
    files_life_time = Config.WEB_APP_FILES_LIFE_TIME

    deleted = 0
    failed = 0
    for file_path in all_files:
        try:
            parent_path, filename = os.path.split(file_path)
            file_timestamp = filename.split("_")[0]
            if time() - int(file_timestamp) > files_life_time:
                os.remove(file_path)
                deleted += 1
        except Exception as e:
            logger.error(f"Error deleting {file_path}, {e}")
            failed += 1
    
    logger.info(f"Deleted: {deleted}, failed: {failed}")


if __name__ == "__main__":
    sztarte()