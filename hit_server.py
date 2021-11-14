import requests
import cv2

BASE_IP = 'http://127.0.0.1:5000/'
url = 'post_image'

img = cv2.imread('CBSD68-dataset/CBSD68/original_png/0005.png')
_, data = cv2.imencode('.png',img)

response = requests.post(BASE_IP + url, data=data.tobytes())
print(response.json())