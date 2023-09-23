import cv2
import string
#输入图像640，480，保存图像为640，480，图像名称cap1-开始从cap1-1.jpg ~ cap1-$.jpg

img_path = "../image/"
win_n="input_image"
cv2.namedWindow(win_n)
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("count open camera")
    exit()
i=0
while(True):
    i=i+1
    ret,frame = cap.read()
    frame=cv2.resize(frame,(640,480),interpolation=cv2.INTER_LINEAR)
    if not ret:
        print("count read image")
        break
    cv2.imshow(win_n,frame)
    cv2.imwrite(img_path+"cap2_{}.jpg".format(i),frame)
    print("image saved:{}".format(i))
    if cv2.waitKey(500) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()