from flask import Flask, jsonify, request
import cv2
import numpy as np
import os, io, base64
from test import predict
from PIL import Image

app = Flask(__name__)

@app.route("/")
def root():
    return jsonify({
        "Project": "Color Image Denoising using PixelRL",
        "Authors": "Mathan CS, Mark McMillian, Gilles Aye",
        "University": "Arizona State",
        "Class": "EEE 598: Reinforcement Learning",
        "Working": "200"
    })

@app.route("/post_image", methods = ['POST'])
def post_image():
    try:
        req_data = request
        base64Un = req_data.data[req_data.data.find(b'/9'):]
        tempAb = base64.b64decode(base64Un)
        nparr = np.frombuffer(tempAb, np.uint8)
        img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        # cv2.imshow("Hee", img)
        resp = predict(img)
        
        prediction = Image.fromarray(resp["prediction"].astype("uint8"))
        prediction_rawBytes = io.BytesIO()
        prediction.save(prediction_rawBytes, "JPEG")
        prediction_rawBytes.seek(0)
        prediction_base64 = base64.b64encode(prediction_rawBytes.read())

        noisy = Image.fromarray(resp["noisy"].astype("uint8"))
        noisy_rawBytes = io.BytesIO()
        noisy.save(noisy_rawBytes, "JPEG")
        noisy_rawBytes.seek(0)
        noisy_base64 = base64.b64encode(noisy_rawBytes.read())

        payload = {
            "total_reward": resp["total_reward"],
            "prediction": str(prediction_base64),
            "noisy": str(noisy_base64)
        }
        return jsonify(payload)

    except Exception as e:
        print(e)
        return jsonify({"Error":"Error! Something went wrong."})
  


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port