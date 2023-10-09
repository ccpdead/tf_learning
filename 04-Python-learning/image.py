import cv2
import pandas as pd

image_path = "/home/jhr/depth_image1.png"
csv_path="/home/jhr/depth.csv"

img_input=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
data = pd.read_csv(csv_path)
print(data)
