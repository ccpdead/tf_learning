#!/home/jhr/anaconda3/bin/python3
import cv2
import numpy as np
import tensorflow as tf

name = {0: "cup", 1: "gcup", 2: "bowl", 3: "plate", 4: "spoon"}
# 加载模型，读取签名
# new_model = tf.saved_model.load("../Yolo/yolov5/runs/train/exp/weights/last_saved_model")
new_model = tf.saved_model.load("/home/jhr/Program/TensorFlow/05-Yolo/model/last_saved_model-2")
model_signature = new_model.signatures["serving_default"]


def compute(img):
    img = tf.convert_to_tensor(img)
    # img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [640, 640])
    img = img / 255.0
    dataset = tf.constant(img)  # IMG转化为Tensor
    dataset = np.expand_dims(dataset, 0)
    resout = model_signature(dataset)  # 模型预测
    resout = resout["output_0"]
    resout = np.array(resout)
    resout[0][..., :4] *= [640, 480, 640, 480]
    return resout


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2.0  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2.0  # bottom right y
    return y


def img_show(Rect, src):
    # print("show image.........................")
    # src = cv2.imread(img_path)
    for i in range(len(Rect)):
        x = int(Rect[i, 0])
        y = int(Rect[i, 1])
        x2 = int(Rect[i, 2])
        y2 = int(Rect[i, 3])
        # print(Rect[i,:])
        cv2.rectangle(src, (x, y), (x2, y2), (0, 255, 0), 3)
        cv2.putText(src, name[int(Rect[i, -1])], (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_4)
        cv2.putText(src, "{:.2}%".format(Rect[i, -2]), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    cv2.LINE_4)
    cv2.imshow("Rect", src)
    # cv2.waitKey(0)


def non_max_suppression(resout, conf_thres=0.5, iou_thres=0.8, mi=10):
    max_wh = 7680
    max_nms = 30000
    max_det = 300
    bs = resout.shape[0]  # batch_size
    nc = resout.shape[2] - 5  # number of classes
    xc = resout[..., 4] > conf_thres  # condidates
    output = [tf.zeros((0, 6))] * bs
    for xi, x in enumerate(resout):
        x = x[xc[xi]]
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])  # 计算四个边框
        mask = x[:, mi:]

        conf = tf.reduce_max(x[:, 5:mi], axis=1, keepdims=True)  # 计算每一列最大值
        j = np.expand_dims(tf.math.argmax(x[:, 5:mi], axis=1), -1)  # 保存每列最大值的索引

        concatenated = tf.concat([box, conf, j, mask], axis=1)
        conf_mask = tf.reshape(conf, [-1]) > conf_thres  # 展平矩阵，并与conf_thres比较
        x = tf.boolean_mask(concatenated, conf_mask, axis=0)  # center_x, center_y, width,

        sorted_indices = tf.argsort(x[:, 4], direction="DESCENDING")
        x = tf.gather(x, sorted_indices[:max_nms])

        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = tf.image.non_max_suppression(boxes, scores, iou_threshold=iou_thres, max_output_size=15)
        i = i[:max_det]  # limit detections
        output[xi] = tf.gather(x, i)
    img_show(output[0], frame)


cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("count open camera")
    exit()
while (True):
    ret, frame = cap.read()
    # cv2.imshow("input",frame)
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    resout = compute(frame)
    non_max_suppression(resout)
    # cv2.imshow('frame', gray)
    if cv2.waitKey(30) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
