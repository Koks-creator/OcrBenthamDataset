from glob import glob
from typing import Tuple, List
import numpy as np
from dataclasses import dataclass
from tensorflow.types.experimental import TensorLike
import tensorflow as tf
from tensorflow import keras
import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config
from custom_logger import CustomLogger

logger = CustomLogger(
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL,
    log_file_name=fr"{Config.ROOT_PATH}/logs/logs.log"
).create_logger()


@dataclass
class OcrPredictor:
    model_folder_path: str

    def __post_init__(self) -> None:
        logger.info(f"{self.model_folder_path=}")

        self.model_file = glob(f"{self.model_folder_path}/*.h5")[0]
        self.num2char_file = glob(f"{self.model_folder_path}/*.pkl")[0]
        logger.info(f"{self.model_file=}")
        logger.info(f"{self.num2char_file=}")

        self.model = keras.models.load_model(self.model_file)
        with open(self.num2char_file, "rb") as n2c_f:
            self.num_to_char = pickle.load(n2c_f)
        logger.info("Model loaded")

    @staticmethod
    def distortion_free_resize(image: TensorLike, img_size: Tuple[int, int]) -> TensorLike:
        logger.info(f"distortion_free_resize: {image.shape=}, {img_size=}")
        w = img_size[1]
        h = img_size[0]
        # Resize with aspect ratio preserved.
        image = tf.image.resize(
            image, size=(h, w), preserve_aspect_ratio=True
        )

        # Calculate how much padding (height & width) is needed
        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]

        # Split that padding equally on top/bottom and left/right
        # (If the needed pad is odd, we add the "extra" pixel on top/left)
        if pad_height % 2 != 0:
            half = pad_height // 2
            pad_height_top = half + 1
            pad_height_bottom = half
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            half = pad_width // 2
            pad_width_left = half + 1
            pad_width_right = half
        else:
            pad_width_left = pad_width_right = pad_width // 2

        # Apply symmetric padding
        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ]
        )
        return image

    def decode_batch_predictions(self, pred: TensorLike) -> List[str]:
        logger.info(f"decode_batch_predictions: {pred.shape=}")
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search.
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

        # Iterate over the results and get back the text.
        output_text = []

        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)

        return output_text
    
    def get_predictions(self, images: List[TensorLike], img_size: Tuple[int, int]) -> List[str]:
        logger.info(f"get_predictions: {len(images)}, {img_size=}")
        images_to_process = []
        for image in images:
            image = self.distortion_free_resize(image, img_size)
            image = tf.cast(image, tf.float32) / 255.0
            images_to_process.append(image)
        
        preds = self.model.predict(np.array(images_to_process))
        pred_texts = self.decode_batch_predictions(preds)

        return pred_texts


if __name__ == "__main__":
    ocr_predictor = OcrPredictor(
            model_folder_path=f"{Config.MODELS_FOLDER_PATH}{Config.MODEL_NAME}"
        )
    images = []
    images_paths = [
        "images/chuj.png"
        # "images/aje.png",
        # "images/niespodzianka.png",
        # "images/r06-111-00-00.png",
        # "images/Screenshot_4.png"
    ]
    for image_path in images_paths:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, 1)
        print(image.shape)
        # print(image[-1])
        images.append(image)

    print(ocr_predictor.get_predictions(
        images=images,
        img_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)
    ))
    
