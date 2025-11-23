import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import uvicorn

from ocr_api import app
from config import Config

def run_api() -> None:
    uvicorn.run(app, 
                host=Config.API_HOST, 
                port=Config.API_PORT,
                log_config=Config().get_uvicorn_logger()
            )

if __name__ == "__main__":
    run_api()