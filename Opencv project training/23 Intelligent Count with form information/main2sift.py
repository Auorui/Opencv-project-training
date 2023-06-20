import cv2
import numpy as np

def count_round_sticks(input_image, template_image):
    """
    :param input_image: 输入图像
    :param template_image: 模板图像
    :return: 返回特征点匹配数量和绘制的匹配结果图像
    """
    input_img = cv2.imread(input_image, 0)
    template_img = cv2.imread(template_image, 0)
    # 创建SIFT对象
    sift = cv2.SIFT_create()
    # 在输入图像和模板图像上检测并计算特征点和描述符
    input_keypoints, input_descriptors = sift.detectAndCompute(input_img, None)
    template_keypoints, template_descriptors = sift.detectAndCompute(template_img, None)
    # 创建FLANN匹配器
    flann = cv2.FlannBasedMatcher()
    # 使用k近邻算法进行特征匹配
    matches = flann.knnMatch(template_descriptors, input_descriptors, k=2)
    # 应用比值测试来筛选匹配项
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    matched_image = cv2.drawMatches(template_img, template_keypoints, input_img, input_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), matched_image

input_image = "./photodata\image003.png"
template_image = "./photodata\image001.png"
count, matched_image = count_round_sticks(input_image, template_image)
print(count)
cv2.imshow("Matched Image", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
