import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import io
import requests
import cv2

from config import Config

files = []
test_files = [fr"{Config.TEST_DATA_FOLDER}/gunsondupaxd_0_414.jpg"]
for file_path in test_files:
    with open(file_path, "rb") as f:
        file_content = f.read()
    files.append(
        ("files", (file_path, io.BytesIO(file_content), "image/png"))
    )

req = requests.post(f"http://{Config.API_HOST}:{Config.API_PORT}/get_ocr_res", files=files)
resp = req.json()

print(resp["result"])
images = [cv2.imread(test_files[0])]

for img_ind, lines_list in resp["result_detailed"].items():
    img_ind = int(img_ind)
    for line in lines_list.values():
        x1, y1, x2, y2 = line["bbox"]
        cv2.rectangle(images[img_ind], (x1, y1), (x2, y2), (200, 0, 200), 1, 1)
        cv2.putText(images[img_ind], line["ocr_res"], (x1, y1-1), cv2.FONT_HERSHEY_PLAIN, .7, (200, 0, 200), 1)
    cv2.imshow("res", images[img_ind])
    cv2.waitKey(0)
