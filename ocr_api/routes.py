import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import List, Tuple, Dict
from pydantic import BaseModel
import cv2
import numpy as np
import asyncio
from fastapi import HTTPException, status, UploadFile, File, Body

from ocr_api import app, logger, Config, document_ocr

class OcrDetail(BaseModel):
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    ocr_res: str

class OcrResponse(BaseModel):
    result: List[str]
    result_detailed: Dict[int, Dict[int, OcrDetail]]

class HealthResponse(BaseModel):
    status: str

@app.get("/")
async def alive():
    return "Hello, I'm alive :) https://www.youtube.com/watch?v=9DeG5WQClUI"

@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    return HealthResponse(status="all green")

@app.post("/get_ocr_res", response_model=OcrResponse)
async def get_ocr_res(files: List[UploadFile] = File(...),
                      conf: float = Body(.2),
                      iou: float = Body(.35),
                      augment: bool = Body(True),
                      agnostic_nms: bool = Body(True)):
    
    logger.info(f"Uploaded {len(files)} files")
    if len(files) > Config.MAX_IMAGE_FILES:
        logger.error(f"(status code 400) Max number of files is {Config.MAX_IMAGE_FILES}")
        raise HTTPException(
            status_code=400,
            detail=f"Max number of files is {Config.MAX_IMAGE_FILES}"
        )
    try:
        logger.debug("Preparing images")

        images = []
        for file in files:
            content = await file.read()
            nparr = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            images.append(image)

        logger.debug("Preparing predictions")
        result, result_detailed = await asyncio.to_thread(document_ocr.ocr_full_flow,
                                                          images, conf,
                                                          iou, augment,
                                                          agnostic_nms)
        logger.debug(f"Predictions: {len(result)=}, {len(result_detailed)=}")

        return OcrResponse(
            result=result,
            result_detailed=result_detailed
        )
    except HTTPException as http_ex:
            logger.error(f"HTTPException {http_ex}")
            raise http_ex
    except Exception as e:
        logger.error(f"(status code 500) Internal server error {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error {e}")
