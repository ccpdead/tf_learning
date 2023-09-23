import tensorflow as tf
import numpy as np
import cv2
import torch
import torchvision


img_path="C:/Users/yeolume/Pictures/01-image/resize_126.jpg"
name = {0:"cup",1:"gcup",2:"bowl",3:"plate",4:"spoon"}
# 加载模型，读取签名
new_model=tf.saved_model.load("C:/Users/yeolume/source/TensorFlow/Yolo/yolov5/runs/train/exp3/weights/last_saved_model")
model_signature=new_model.signatures["serving_default"]

img=tf.io.read_file(img_path)
img=tf.image.decode_jpeg(img,channels=3)
img=tf.image.resize(img,[640,640])
img=img/255.0


dataset=tf.constant(img)
print("shape1 ",dataset.shape)
dataset=np.expand_dims(dataset,0)
print("shape2 ",dataset.shape)

resout=model_signature(x=dataset)#模型预测
resout=resout["output_0"]
resout=resout.numpy()
resout[0][...,:4]*=[400,400,400,400]

def img_show(Rect):
    print("show image.........................")
    src = cv2.imread(img_path)
    for i in range(len(Rect)):
        x=int(Rect[i,0])
        y=int(Rect[i,1])
        x2=int(Rect[i,2])
        y2=int(Rect[i,3])
        print(Rect[i,:])
        cv2.rectangle(src,(x,y),(x2,y2),(0,255,0),3)
        cv2.putText(src,name[int(Rect[i,-1])],(x,y+15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_4)
        cv2.putText(src,"{:.2}%".format(Rect[i,-2]),(x,y+30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_4)
    cv2.imshow("Rect",src)
    cv2.waitKey(0)

def xywh2xyxy(x):
    y=np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0 # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2.0  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2.0  # bottom right y
    return torch.tensor(y)

def non_max_suppression(resout,conf_thres=0.25,iou_thres=0.45,mi=10):
    max_wh = 7680
    max_nms = 30000
    max_det = 300
    bs = resout.shape[0]  # batch_size
    nc = resout.shape[2] - 5  # number of classes
    xc = resout[..., 4] > conf_thres  # condidates
    output = [tf.zeros((0, 6))] * bs
    for xi,x in enumerate(resout):
        x=torch.tensor(x)#抓换为tensor对象
        x=x[xc[xi]]
        x[:,5:]*=x[:,4:5]
        box=xywh2xyxy(x[:,:4])#计算四个边框
        mask=x[:,mi:]
        conf, j = x[:, 5:mi].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS 非最大值抑制
        i = i[:max_det]  # limit detections
        output[xi] = x[i]


    img_show(output[0])
    return output

if __name__ == '__main__':
    print(non_max_suppression(resout))