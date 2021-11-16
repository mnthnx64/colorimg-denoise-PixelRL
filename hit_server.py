import requests
import cv2
import base64

BASE_IP = 'http://127.0.0.1:5000/'
url = 'post_image'

img = cv2.imread('/Users/mx98/Downloads/222.jpeg',1)
_, data = cv2.imencode('.jpg',img)

response = requests.post(BASE_IP + url, data=base64.b64encode(data))
print(response.json())