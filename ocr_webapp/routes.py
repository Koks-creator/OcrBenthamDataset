import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import requests
import io
import time
import cv2
from flask import render_template, session, redirect, url_for, flash, request

from config import Config
from ocr_webapp import app, forms


@app.route("/", methods=["GET", "POST"])
def home():
    try:
        form = forms.MainForm()
        image_filenames = []
        results = []

        if not app.config["TESTING"]:
            val_mode = 0
            form_validation = form.validate_on_submit()
        else:
            val_mode = 1
            form_validation = request.method == 'POST'

        if form_validation:
            # this if is made to prevent errors when using not allowed file extension in testing mode
            if val_mode == 1:
                for img in form.images.data:
                    if os.path.splitext(img.filename)[1] not in (".png", ".jpg", ".jpeg"):
                        return {"Status": "File should be png or jpg"}, 400
            
            app.logger.info(f"Uploaded {len(form.images.data)} files, preparing...")
            file_contents = []
            for image_file in form.images.data:
                content = image_file.read()
                timestamp = int(time.time())
                filename = f"{timestamp}_{image_file.filename}"
                file_path = rf"{Config.WEB_APP_TEMP_UPLOADS_FOLDER}\{filename}"

                with open(file_path, "wb") as f:
                    f.write(content)
                image_filenames.append(filename)
                file_contents.append(content)
                
            files = []
            for file_content in file_contents:
                files.append(
                    ("files", (file_path, io.BytesIO(file_content), "image/png"))
                )

            app.logger.info("Making request to api")
            req = requests.post(f"http://{Config.API_HOST}:{Config.API_PORT}/get_ocr_res", files=files)
            resp = req.json()

            for img_ind, lines_list in resp["result_detailed"].items():
                img_ind = int(img_ind)
                img_path = rf"{Config.WEB_APP_TEMP_UPLOADS_FOLDER}\{image_filenames[img_ind]}"
                img = cv2.imread(img_path)
                for line in lines_list.values():
                    x1, y1, x2, y2 = line["bbox"]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 0, 200), 1, 1)
                    cv2.putText(img, line["ocr_res"], (x1, y1-1), cv2.FONT_HERSHEY_PLAIN, .7, (200, 0, 200), 1)
                    cv2.imwrite(img_path, img)
            results = resp["result"]
            app.logger.info(f"Predictions: {len(resp['result'])=}, {len(resp['result_detailed'])=}")
        
        json_res = {}
        if len(results) == len(image_filenames):
            for res, filename in zip(results, image_filenames):
                json_res[filename] = res
        else:
            app.logger.error(f"Co tu sie odjebało? {len(results)=} != { len(image_filenames)=}")

        #     app.logger.info(f"{predictions=}")
        # if image_datas:
        #     res = zip(image_datas, predictions)
        # else:
        #     res = []
        # res_to_save = {
        #     "Filename": "",
        #     "Prediction": ""
        # }
        return render_template("home.html", form=form, results=results, image_filenames=image_filenames, json_res=json_res)
    except Exception as e:
        app.logger.error(f"Unknown error: {e}")
        return redirect(url_for("error_page", error=e, status_code=500))
    

@app.route("/error", methods=["GET"])
def error_page():
    error_msg = request.args.get("error", "Unknown error occured")
    error_status_code = request.args.get("status_code", 500)
    try:
        error_status_code = int(error_status_code)
    except ValueError:
        error_status_code = 500
    
    return render_template("error_page.html", status_code=500, error_text=error_msg)
