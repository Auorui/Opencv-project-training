import cv2
import numpy as np
import os
import pandas as pd

class VideoCap():
    """
    自定义的摄像头调取类
    """
    def CapInit(self,mode=0,w=640,h=480,l=150):
        """
        摄像头初始化函数
        :param mode: 0-web,1-ex camera
        :param w: 窗口宽度，默认为640
        :param h: 窗口长度，默认为480
        :param l: 亮度，默认为150
        """
        self.cap = cv2.VideoCapture(mode)
        self.mode=mode
        self.cap.set(3, w)
        self.cap.set(4, h)
        self.cap.set(10, l)
    def read(self):
        """
        摄像头读取函数
        :return: 仅仅返回图片
        """
        _, img = self.cap.read()
        return img

def readPath(folder_path):
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_files.append(os.path.join(folder_path, filename))
    return image_files

def ImagePath(recongnitionMode):
    if len(recongnitionMode)>10:
        save_dir = "./Pic/img"
        excel_path = "./Pic/data.xlsx"
        return save_dir, excel_path
    else:
        save_dir = "./Video/img"
        excel_path = "./Video/data.xlsx"
        return save_dir, excel_path

def Bbox_img(img, bbox_scale=0.5, bbox_color=(0, 0, 255)):
    """
    默认在图像中央显示红框
    :param img: 输入图像
    :param bbox_scale: 占图像的比例，默认为50%
    :param bbox_color: 边框颜色
    :return: 返回含有边框的图像
    """
    height, width = img.shape[:2]
    box_width = int(width * bbox_scale)
    box_height = int(height * bbox_scale)

    start_x = int((width - box_width) / 2)
    start_y = int((height - box_height) / 2)
    end_x = start_x + box_width
    end_y = start_y + box_height
    postion=[(start_x, start_y),(end_x, end_y)]
    img_with_box = img.copy()
    cv2.rectangle(img_with_box, postion[0], postion[1], bbox_color, 2)
    return postion,img_with_box

def maskBbox(img, position):
    """
    创建遮罩,对框外的图像进行高斯模糊处理
    :param img: 输入图像
    :param position: Bbox_img的返回值
    :return: 受过遮罩保护的图像，对非遮罩区域进行高斯模糊处理
    """
    mask = np.zeros_like(img)
    cv2.rectangle(mask, position[0], position[1], (255, 255, 255), -1)
    blurred_img = cv2.GaussianBlur(img, (57, 57), 0)
    img_with_blur = np.where(mask == 0, blurred_img, img)
    return img_with_blur


def onTrackbarChange(scale_value):
    """
    轨迹栏函数，主要用于全局的定义bbox_scale, brightness_factor
    """
    global bbox_scale, brightness_factor
    bbox_scale = scale_value / 1000.0 + 0.1
    brightness_factor = scale_value / 100.0 + 0.5

def empty(a):
    """
    轨迹栏函数
    """
    pass

def preProcessing(img):
    """
    预处理阶段
    :param img: 图像
    :return: 经过了高斯、canny，膨胀、闭运算操作
    """
    imgPre = cv2.GaussianBlur(img, (5, 5), 3)
    thresh1 = cv2.getTrackbarPos("Threshold1", "Settings")
    thresh2 = cv2.getTrackbarPos("Threshold2", "Settings")
    imgPre = cv2.Canny(imgPre, thresh1, thresh2)
    kernel = np.ones((3, 3), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)

    return imgPre

def findContours(img, imgPre, minArea=1000, sort=True, filter=0, c=(255, 0, 0)):
    """
    寻找物体的边缘信息，采用框选的方式，对于物体的表现更加的准确，经过测试对于轮廓的检测效果很差
    :param img: 图片
    :param imgPre: 预处理后的图片
    :param minArea: 最小面积阈值，用于过滤小面积的边缘。默认值为 1000
    :param sort: 是否按照面积进行排序。如果为 True，按照面积从大到小排序。默认值为 True
    :param filter: 过滤多边形的边数。如果为 0，则不进行过滤；如果为其他正整数，则只保留边数等于该值的多边形。默认值为 0
    :param c: 边框和点的颜色，默认为红色
    :return:
    """
    conFound = []
    imgContours = img.copy()
    contours, hierarchy = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == filter or filter == 0:
                x, y, w, h = cv2.boundingRect(approx)
                cx, cy = x + (w // 2), y + (h // 2)
                if 3000<= area <= 16000:
                    cv2.rectangle(imgContours, (x, y), (x + w, y + h), c, 2)
                    cv2.circle(imgContours, (x + (w // 2), y + (h // 2)), 5, c, cv2.FILLED)
                    conFound.append({"cnt": cnt, "area": area, "bbox": [x, y, w, h], "center": [cx, cy]})
                    total+=1
    if sort:
        conFound = sorted(conFound, key=lambda x: x["area"], reverse=True)
    return imgContours, conFound, total

def stackImages(scale,imgArray):
    """
    :param scale: 规格
    :param imgArray: 图片形式列表
    :return: 堆叠后的图片
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def Adjusted_image(img,brightness_factor = 1.5):
    """
    :param img: 图片
    :param brightness_factor: 亮度调整因子
    :return:
    """
    image_float = img.astype(np.float32)
    adjusted_image = image_float * brightness_factor
    adjusted_image = np.clip(adjusted_image, 0, 255)
    adjusted_image = adjusted_image.astype(np.uint8)
    return adjusted_image

def excelmation(excel_path,data):
    """
    :param excel_path: 电子表格路径
    :param data: 写入数据类型
    :return:
    """
    if not os.path.exists(excel_path):
        df = pd.DataFrame(columns=["Time", "Image Name", "Total"])
        df.to_excel(excel_path, index=False, sheet_name="Data")
    df = pd.DataFrame(data, columns=["Time", "Image Name", "Total"])
    df.to_excel(excel_path, index=False, sheet_name="Data")


def drawContour(img, imgPre, minArea=15, show=True):
    """
    :param img: 图片
    :param imgPre: 预处理图片
    :param minArea: 最小面积
    :return:
    """
    imgContours, conFound, total = findContours(img, imgPre, minArea)
    imgStacked = stackImages(1, [img, imgContours])
    cv2.putText(imgStacked, f"Count: {total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if show:
        cv2.imshow("Settings", imgStacked)
    cv2.putText(imgContours, f"Count: {total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return imgStacked,total,imgContours
