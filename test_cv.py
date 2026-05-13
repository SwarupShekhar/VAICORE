import cv2
import numpy as np
import requests
from io import BytesIO

url = "https://s3.amazonaws.com/htx-pub/datasets/images/125245483_152578129892066_7843809718842085333_n.jpg" # dummy or we can use local
# Wait, let's use the local file if possible. What is the URL of the image they uploaded?
# Actually, let's just write a generic test to see if cv2 works.
print("OpenCV is ready!")
