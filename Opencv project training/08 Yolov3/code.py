import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.2  #the lower it is the more aggressive it will be
print(f"以下是你可以检测的{91}种物品:")

#### LOAD MODEL#######################################
##导入coco.names文件当中的内容
classesFile = "coco.names"                           #
classNames = []                                      #
with open(classesFile, 'rt') as f:                   #
    classNames = f.read().rstrip('\n').split('\n')   #
print(classNames)                                    #
######################################################

#### Model Files###################################################
## 引进我们的模块
modelConfiguration = "yolov3-320.cfg"                             #
modelWeights = "yolov3-320.weights"                               #
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights) #   模型配置和权重配置
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)               #   编写nets点集，首选后端
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)                    #   就要使用CPU提高精度，当然要想获得更高的，就要使用GPU
###################################################################

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]  #我们需要寻找某一个类中，判别最高的
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)  #值是百分比，乘以图像的大小才是正确的像素值
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)  #置信度阈值以及nms的阈值
    # print(indices)
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = cap.read()
    #创建我们的网络，将图像转化为blob的形式，因为这是网络可以理解的形式
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)   #裁剪的结果是Flase,参数是默认值
    net.setInput(blob)    #设置网络前行的通行证
    layersNames = net.getLayerNames()
    # print(layersNames)  #获取所有图层的名字
    # print(net.getUnconnectedOutLayers())   #三个不同的输出层
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    outputs = net.forward(outputNames)
    # print(outputs)
    # print(len(outputs))  #检测是否取到我们需要的那三层,输出为3，True
    outputs=list(outputs)     #输出之前是元组
    # print(type(outputs[0]))   #<class 'numpy.ndarray'>
    ##########################################
    # print(outputs[0].shape)        #(300, 85)  300，1200，4800 is is the number of bounding boxes
    # print(outputs[1].shape)        #(1200, 85)   85含有中心x,y,宽,高，物体的置信度
    # print(outputs[2].shape)        #(4800, 85)   其余的是每个类别的预测概率，也就是value number
    print(outputs[0][0])
    ##########################################
    findObjects(outputs, img)

    cv.imshow('Image', img)
    if cv.waitKey(1) & 0xFF ==27:
        break