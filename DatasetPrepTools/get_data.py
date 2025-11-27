import os
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config

cap = cv2.VideoCapture(rf"C:\Users\table\PycharmProjects\MojeCos\amerykasimulator\Videos\8671565-sd_960_540_25fps.mp4")
class_name = "ajajaj5"
data_dir = Config.TRAIN_IMAGES_FOLDER
while True:
    success, img = cap.read()
    # img = cv2.flip(img, 1)
    if success is False:
        break

    cv2.imshow("Res", img)

    key = cv2.waitKey(10)
    if key == ord("s"):
        print("Saved")
        cv2.imwrite(rf"{data_dir}/{class_name}_{len(os.listdir(data_dir))}.jpg", img)

    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()