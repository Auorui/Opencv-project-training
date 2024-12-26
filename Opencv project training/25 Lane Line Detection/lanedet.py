import cv2
import time
import numpy as np

def Preprocess_images(image, ksize, sigma=None, low_threshold=50, high_threshold=250):
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    gaussianblur_img = cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=sigma)
    gray_img = cv2.cvtColor(gaussianblur_img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, low_threshold, high_threshold)

    return canny_img

def Roi_Mask(gray_image):
    mask = np.zeros_like(gray_image)
    height, width = gray_image.shape
    # 如果效果不是很好, 可以调整此处的比例
    left_bottom = [0, height]
    right_bottom = [width, height]
    left_top = [width / 3, height / 1.5]
    right_top = [width / 3 * 2, height / 1.5]

    vertices = np.array([left_top,
                        right_top,
                        right_bottom,
                        left_bottom
                         ], np.int32)

    cv2.fillPoly(mask, [vertices], (255, ))
    masked_image = cv2.bitwise_and(gray_image, mask)
    return masked_image

def draw_lines(img, lines, color=(0, 255, 255), thickness=2, is_draw_polygon=True):
    left_lines_x = []
    left_lines_y = []
    right_lines_x = []
    right_lines_y = []
    line_y_max = 0
    line_y_min = 999

    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 > line_y_max:
                    line_y_max = y1
                if y2 > line_y_max:
                    line_y_max = y2
                if y1 < line_y_min:
                    line_y_min = y1
                if y2 < line_y_min:
                    line_y_min = y2

                k = ( (y2 - y1) / (1e-6 + x2 - x1))

                if k < -0.3:
                    left_lines_x.append(x1)
                    left_lines_y.append(y1)
                    left_lines_x.append(x2)
                    left_lines_y.append(y2)
                elif k > 0.3:
                    right_lines_x.append(x1)
                    right_lines_y.append(y1)
                    right_lines_x.append(x2)
                    right_lines_y.append(y2)
        # 最小二乘直线拟合
        left_line_k, left_line_b = np.polyfit(left_lines_x, left_lines_y, 1)
        right_line_k, right_line_b = np.polyfit(right_lines_x, right_lines_y, 1)

        cv2.line(img,
                 (int((line_y_max - left_line_b) / left_line_k), line_y_max),
                 (int((line_y_min - left_line_b) / left_line_k), line_y_min),
                 color, thickness)
        cv2.line(img,
                 (int((line_y_max - right_line_b) / right_line_k), line_y_max),
                 (int((line_y_min - right_line_b) / right_line_k), line_y_min),
                 color, thickness)
        zero_img = np.zeros((img.shape), dtype=np.uint8)
        if is_draw_polygon:
            polygon = np.array([
                [int((line_y_max - left_line_b) / left_line_k), line_y_max],
                [int((line_y_max - right_line_b) / right_line_k), line_y_max],
                [int((line_y_min - right_line_b) / right_line_k), line_y_min],
                [int((line_y_min - left_line_b) / left_line_k), line_y_min]
            ])
            cv2.fillConvexPoly(zero_img, polygon, color=(0, 255, 0))

        img = cv2.addWeighted(img, 1, zero_img, beta = 0.2, gamma=0)

    except Exception:
        pass

    return img


class FPS:
    def __init__(self):
        self.pTime = time.time()

    def update(self, img=None, pos=(20, 50), color=(255, 0, 0), scale=3, thickness=3):
        cTime = time.time()
        try:
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            if img is None:
                return fps
            else:
                cv2.putText(img, f'FPS: {int(fps)}', pos, cv2.FONT_HERSHEY_PLAIN,
                            scale, color, thickness)
                return fps, img
        except:
            return 0

def dectet_lane_line(video_path, ksize=3, end_k=27, is_draw_polygon=True):
    cap = cv2.VideoCapture(video_path)
    fps_counter = FPS()
    while True:
        ret, img = cap.read()
        video = Preprocess_images(img, ksize=ksize)
        roi_image = Roi_Mask(video)
        line_img = cv2.HoughLinesP(roi_image,
                                   rho=1,
                                   theta=np.pi / 180,
                                   threshold=15,
                                   minLineLength=40,
                                   maxLineGap=20)

        img = draw_lines(img, line_img, is_draw_polygon=is_draw_polygon)
        fps, img = fps_counter.update(img)
        cv2.imshow('frame', img)

        k = cv2.waitKey(1)
        if k == end_k:
            break

    cap.release()
    cv2.destroyAllWindows()

def dectet_lane_line_ui(
        img,
        ksize,
        minVotesForLine=15,
        minLineLength=40,
        maxLineGap=20,
        is_draw_polygon=True):
    video = Preprocess_images(img, ksize=ksize)
    roi_image = Roi_Mask(video)
    line_img = cv2.HoughLinesP(roi_image,
                               rho=1,
                               theta=np.pi / 180,
                               threshold=minVotesForLine,
                               minLineLength=minLineLength,
                               maxLineGap=maxLineGap)
    img = draw_lines(img, line_img, is_draw_polygon=is_draw_polygon)

    return img

if __name__=="__main__":
    video_path = r"videofiles/video1.mp4"
    dectet_lane_line(video_path)
    # cap = cv2.VideoCapture(video_path)
    #
    # while True:
    #     ret, img = cap.read()
    #     video = Preprocess_images(img, ksize=3)
    #     roi_image = Roi_Mask(video)
    #     line_img = cv2.HoughLinesP(roi_image,
    #                                rho=1,
    #                                theta=np.pi/180,
    #                                threshold=15,
    #                                minLineLength=40,
    #                                maxLineGap=20
    #                                )
    #     img = draw_lines(img, line_img, is_draw_polygon=True)
    #     cv2.imshow('frame', img)
    #     k = cv2.waitKey(1)
    #
    #     if k == 27:
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()