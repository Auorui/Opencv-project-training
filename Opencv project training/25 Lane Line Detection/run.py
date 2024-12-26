from lane import Ui_Laneline
from lanedet import dectet_lane_line_ui
import sys
import cv2
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QTimer, QCoreApplication
import qimage2ndarray


class LaneDetectionShow(QMainWindow, Ui_Laneline):
    def __init__(self, parent=None):
        super(LaneDetectionShow, self).__init__(parent)
        self.setupUi(self)
        self.PrepParameters()
        self.CallBackFunctions()
        self.InitUI()  # 初始化UI组件的额外设置

        self.ksize = 3
        # self.min_votes_for_line = 15
        # self.min_line_length=40,
        # self.max_line_gap=20,
        # self.is_draw_polygon=True,

    def PrepParameters(self):
        self.PFilePath = r'./videofiles\video1.mp4'
        self.PFilePathLiEd.setText(self.PFilePath)
        video_path = self.PFilePath
        self.camera = cv2.VideoCapture(video_path)
        if not self.camera.isOpened():
            print("Error: Could not open video.")

    def CallBackFunctions(self):
        """将lineedit与button连接"""
        self.PFilePathBt.clicked.connect(self.SetPFilePath)
        self.RunBt.clicked.connect(self.StartVideo)

    def SetPFilePath(self):
        file_filter = "Video Files (*.mp4 *.avi *.mov *.mkv)"
        options = QFileDialog.Options()
        start_dir = '.'
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", start_dir, file_filter, options=options)
        if file_name:
            self.PFilePathLiEd.setText(file_name)
            self.PFilePath = file_name
            self.camera.release()
            self.camera = cv2.VideoCapture(file_name)
            if not self.camera.isOpened():
                print("Error: Could not open selected video.")

    def DispImg(self):
        if self.Image is not None:
            self.Image = dectet_lane_line_ui(
                self.Image,
                self.ksize,
                # minVotesForLine=,
                # minLineLength=,
                # maxLineGap=,
                # is_draw_polygon=,
            )
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
            qimg = qimage2ndarray.array2qimage(self.Image)
            self.DispLb.setPixmap(QPixmap.fromImage(qimg))
            self.DispLb.setScaledContents(True)

    def StartVideo(self):
        self.Timer = QTimer()
        self.Timer.timeout.connect(self.TimerOutFun)
        self.Timer.start(30)  # 启动定时器，每30毫秒触发一次

    def TimerOutFun(self):
        success, self.Image = self.camera.read()
        if success:
            self.DispImg()
        else:
            self.Timer.stop()
            print("Video playback finished.")

    def InitUI(self):
        self.RunBt.setEnabled(True)

    def ExitApp(self):
        self.Timer.Stop()
        self.camera.release()
        QCoreApplication.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = LaneDetectionShow()
    ui.show()
    sys.exit(app.exec_())