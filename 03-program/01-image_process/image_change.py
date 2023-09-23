#对预训练的图像进行处理
import random
import cv2
import glob


image_path = glob.glob("image/*.jpg")

for i in range(100):
    i=random.randrange(100)

    src=cv2.imread("{}".format(image_path[i]))
    # src_r=cv2.resize(src,(640,640))#图像大小转换
    # src_r = cv2.rotate(src,cv2.ROTATE_180)#图像翻转
    src_r = cv2.cvtColor(src,cv2.COLOR_BGR2HSV_FULL)#图像颜色更改


    cv2.imshow("rotation",src_r)
    cv2.imwrite("{}".format(image_path[i]),src_r)
    print("{}".format(image_path[i])+" image saved")
    if cv2.waitKey(100) == ord('q'):
        break

cv2.destroyAllWindows()