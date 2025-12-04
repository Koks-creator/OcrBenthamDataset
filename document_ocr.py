from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union, Generator, List
from collections import defaultdict
import cv2
import numpy as np


from ocr_predictor import OcrPredictor
from custom_decorators import timeit, log_call
from custom_logger import CustomLogger
from config import Config
from yolo_detector import YoloDetector
from tools import auto_sort_boxes

logger = CustomLogger(logger_log_level=Config.CLI_LOG_LEVEL,
                      file_handler_log_level=Config.FILE_LOG_LEVEL,
                      log_file_name=fr"{Config.ROOT_PATH}/logs/logs.log"
                      ).create_logger()



@dataclass(frozen=True)
class FrameDetectionData:
    original_frame: np.ndarray
    detection_frame: np.ndarray
    detections_data: Generator


@dataclass
class DocumentOcr:
    ocr_model_path: Union[Path, str]
    yolo_model_path: Union[Path, str]
    yolo_classes_path: Union[Path, str]
    yolo_device: str = "cpu"

    def __post_init__(self) -> None:
        logger.info(
            "Initing DocumentOcr:\n"
            f"{self.ocr_model_path=}\n"
            f"{self.yolo_model_path=}\n"
            f"{self.yolo_classes_path=}\n"
            f"{self.yolo_device=}"
        )
        self.ocr_predictor = OcrPredictor(
            model_folder_path=self.ocr_model_path
            )

        self.yolo_predictor = YoloDetector(
                model_path=self.yolo_model_path,
                classes_path=self.yolo_classes_path
            )
    
    @log_call(logger=logger, log_params=["conf", "iou", "augment", "agnostic_nms"], hide_res=True)
    @timeit(logger=logger)
    def get_yolo_predictions(self, images: List[np.ndarray], conf: float = .2, iou: float = .35,
                             augment: bool = True, agnostic_nms: bool = True
                             ) -> Tuple[List[Generator], 
                                        List[np.ndarray], 
                                        List[np.ndarray]]:
        res = self.yolo_predictor.detect(images=images,
                                         conf=conf,
                                         iou=iou,
                                         augment=augment,
                                         agnostic_nms=agnostic_nms)
        detection_results, detection_frames = map(list, zip(*res)) if res else ([], [])
        detection_generators = [self.yolo_predictor.yield_data(bbox=detection_res) for detection_res in detection_results]

        return detection_generators, detection_frames, images
    
    @log_call(logger=logger, log_params=[""], hide_res=True)
    @timeit(logger=logger)
    def normalize_yolo_predictions(self, detection_generators: List[Generator],
                                   detection_frames: List[np.ndarray],
                                   images: List[np.ndarray]
                                   ) -> List[List[FrameDetectionData]]:
        result = []
        for detection_generator, detection_frame, frame in zip(detection_generators, detection_frames, images):
            result.append(
                FrameDetectionData(
                    original_frame=frame,
                    detection_frame=detection_frame,
                    detections_data=detection_generator
                )
            )
        
        return result
    
    @log_call(logger=logger, log_params=["img_height", "img_height"], hide_res=True)
    @timeit(logger=logger)
    def get_ocr_result(self, normalized_data: List[List[FrameDetectionData]],
                       img_height: int, img_width: int
                       ) -> Tuple[List[str], defaultdict[defaultdict[dict]]]:
        """
        Returns whole text read by ocr from each image (`result`) and more detailed data
        with bbox and ocr result for each line in every image (`result_mapped`)
        """
        result_mapped = defaultdict(dict)
        result = []
        for norm_ind, norm_data in enumerate(normalized_data):
            sorted_boxes = auto_sort_boxes(
                [data[3] for data in norm_data.detections_data],
                strategy='auto',
                debug=True
            )
            result_mapped[norm_ind] = defaultdict(dict)
            cropped_lines = []
            for ind, box in enumerate(sorted_boxes):
                x1, y1, x2, y2 = box
                cropped_line = norm_data.original_frame[y1:y2, x1:x2]
                
                cropped_line_grey = cv2.cvtColor(cropped_line, cv2.COLOR_BGR2GRAY)
                cropped_line_grey = np.expand_dims(cropped_line_grey, axis=-1)
                cropped_lines.append(cropped_line_grey)

                result_mapped[norm_ind][ind]["bbox"] = (x1, y1, x2, y2)

            pred = self.ocr_predictor.get_predictions(images=cropped_lines,
                                                      img_size=(img_height, img_width)
                                                      )
            for ind, p in enumerate(pred):
                result_mapped[norm_ind][ind]["ocr_res"] = p
            result.append("\n".join(pred))
        
        return result, result_mapped
    
    @log_call(logger=logger, log_params=["conf", "iou", "augment", "agnostic_nms", "img_height", "img_width"], hide_res=True)
    @timeit(logger=logger)
    def ocr_full_flow(self, images: List[np.ndarray], conf: float = .2, iou: float = .35,
                      augment: bool = True, agnostic_nms: bool = True,
                      img_height: int = Config.IMAGE_HEIGHT, img_width: int = Config.IMAGE_WIDTH
                      ) -> Tuple[List[str], defaultdict[defaultdict[dict]]]:
        detection_generators, detection_frames, images = self.get_yolo_predictions(
            images=images, conf=conf, iou=iou,
            augment=augment, agnostic_nms=agnostic_nms
            )
        normalized_data = self.normalize_yolo_predictions(detection_generators, detection_frames, images)
        res = self.get_ocr_result(normalized_data=normalized_data,
                                    img_height=img_height,
                                    img_width=img_width)
        
        return res
    

if __name__ == "__main__":
    document_ocr = DocumentOcr(
        ocr_model_path=f"{Config.OCR_MODELS_FOLDER_PATH}{Config.OCR_MODEL_NAME}",
        yolo_model_path=f"{Config.YOLO_MODELS_FOLDER_PATH}{Config.YOLO_MODEL_NAME}",
        yolo_classes_path=Config.YOLO_CLASSES_FILE
    )

    img = cv2.imread(rf"{Config.TEST_DATA_FOLDER}/gunsondupaxd_0_418.jpg")
    img2 = cv2.imread(rf"{Config.TEST_DATA_FOLDER}/gunsondupaxd_0_430.jpg")
    img3 = cv2.imread(rf"{Config.TEST_DATA_FOLDER}/gunsondupaxd_0_431.jpg")
    img4 = cv2.imread(rf"C:\Users\table\PycharmProjects\MojeCos\ocr_dwa\train_data\images\train\gunsondupaxd_0_10.jpg")
    images = [img4]
    res, res_mapped = document_ocr.ocr_full_flow(images=images)
    # print(res_mapped)
    for img_ind, lines_list in res_mapped.items():
        for line in lines_list.values():
            x1, y1, x2, y2 = line["bbox"]
            cv2.rectangle(images[img_ind], (x1, y1), (x2, y2), (200, 0, 200), 1, 1)
            cv2.putText(images[img_ind], line["ocr_res"], (x1, y1-1), cv2.FONT_HERSHEY_PLAIN, .7, (200, 0, 200), 1)
        cv2.imshow("xd", images[img_ind])
        cv2.waitKey(0)
