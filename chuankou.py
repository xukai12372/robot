import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QWidget, QMessageBox, QProgressBar, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QSize
from ultralytics import YOLO
from PIL import Image, ImageDraw
import time
import sys
import threading

class ArmPoseProcessor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.connection_rules = [(0, 1), (1, 2)]  # 关键点连接规则

    def process_single_image(self, image_path):
        """处理单张图片并返回标注结果"""
        try:
            results = self.model.predict(source=image_path, conf=0.2)
            if not results or results[0].keypoints is None:
                return None, "未检测到关键点"
            
            # 处理结果
            processed_img = self._process_result(results[0], image_path)
            return processed_img, "处理成功"
        except Exception as e:
            return None, f"处理失败: {str(e)}"

    def process_frame(self, frame):
        """处理单个视频帧"""
        try:
            # 转换颜色空间从BGR到RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 执行预测
            results = self.model.predict(source=rgb_frame, conf=0.2)
            
            if not results or results[0].keypoints is None:
                return frame, "未检测到关键点"
            
            # 处理结果
            processed_img = self._process_result(results[0], frame)
            return processed_img, "检测成功"
        except Exception as e:
            return frame, f"处理失败: {str(e)}"

    def process_images_to_video(self, input_folder, output_video, fps=30):
        """处理图片序列并生成视频"""
        # 获取并排序输入图片
        image_files = sorted([
            f for f in os.listdir(input_folder) 
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        
        success_count = 0
        size = None
        frame_array = []

        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(input_folder, image_file)
            
            try:
                # 执行预测
                results = self.model.predict(source=image_path, conf=0.2)
                
                # 处理结果
                processed_img = self._process_result(results[0], image_path)
                if processed_img is not None:
                    success_count += 1
                    
                    # 收集视频帧
                    if size is None:
                        size = (processed_img.shape[1], processed_img.shape[0])
                    frame_array.append(processed_img)
                    
            except Exception as e:
                print(f"处理失败 {image_file}: {str(e)}")
                # 使用原始图片作为替代
                original_img = cv2.imread(image_path)
                if original_img is not None:
                    if size is None:
                        size = (original_img.shape[1], original_img.shape[0])
                    frame_array.append(original_img)

        # 生成视频
        if size and frame_array:
            self._create_video(frame_array, output_video, size, fps)
            return True, f"视频生成成功 (基于 {success_count}/{len(image_files)} 有效帧)"
        else:
            return False, "无有效帧生成视频"

    def _process_result(self, result, img):
        """处理单张图片结果"""
        if result.keypoints is None:
            print("未检测到关键点")
            return None
            
        # 获取关键点
        keypoints = result.keypoints.xy[0].cpu().numpy()
        
        # 基础可视化
        if isinstance(img, str):  # 如果是文件路径
            img_array = result.plot(kpt_line=True, kpt_radius=6)
            img_pil = Image.fromarray(img_array[..., ::-1])
        else:  # 如果是numpy数组
            img_array = result.plot(kpt_line=True, kpt_radius=6)
            img_pil = Image.fromarray(img_array[..., ::-1])
        
        draw = ImageDraw.Draw(img_pil)
        
        # 绘制自定义连线
        valid_points = []
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:  # 有效点检查
                valid_points.append((int(x), int(y)))
        
        # 只连接有效点
        for start, end in self.connection_rules:
            if start < len(valid_points) and end < len(valid_points):
                draw.line([valid_points[start], valid_points[end]], 
                         fill=(255, 0, 0), width=3)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _create_video(self, frames, output_path, size, fps):
        """从帧序列生成视频"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        for frame in frames:
            out.write(frame)
            
        out.release()

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频播放器")
        self.setGeometry(100, 100, 800, 600)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        
        self.play_button = QPushButton("播放")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.stop_button.clicked.connect(self.stop_video)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        self.video_capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
    def load_video(self, video_path):
        self.video_capture = cv2.VideoCapture(video_path)
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.timer.start(int(1000 / self.fps))
        
    def play_video(self):
        if self.video_capture:
            self.timer.start(int(1000 / self.fps))
            
    def pause_video(self):
        self.timer.stop()
        
    def stop_video(self):
        self.timer.stop()
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_capture.read()
            if ret:
                self.display_frame(frame)
                
    def update_frame(self):
        if self.video_capture:
            ret, frame = self.video_capture.read()
            if ret:
                self.display_frame(frame)
            else:
                self.stop_video()
                
    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))
            
    def closeEvent(self, event):
        if self.video_capture:
            self.video_capture.release()
        event.accept()

class CameraThread(threading.Thread):
    def __init__(self, processor, display_callback):
        super().__init__()
        self.processor = processor
        self.display_callback = display_callback
        self.running = False
        self.cap = None
        
    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)  # 0表示默认摄像头
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # 处理帧
            processed_frame, _ = self.processor.process_frame(frame)
            
            # 回调显示
            self.display_callback(processed_frame)
            
            # 稍微延迟一下
            time.sleep(0.03)
            
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

class ArmPoseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("机械臂姿态关键点检测系统")
        self.setGeometry(100, 100, 1000, 800)
        
        # 固定窗口大小
        # self.setFixedSize(1000, 800)
        
        self.processor = ArmPoseProcessor("best.pt")
        self.current_image = None
        self.video_path = None
        self.video_player = None
        self.camera_thread = None
        
        self.init_ui()
        
    def init_ui(self):
        # 创建主部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # 图像显示区域 - 设置固定大小
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(800, 600)  # 设置最小尺寸
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.single_image_btn = QPushButton("添加单张图片")
        self.multiple_images_btn = QPushButton("添加多张图片")
        self.camera_btn = QPushButton("打开摄像头")
        self.stop_camera_btn = QPushButton("关闭摄像头")
        self.save_image_btn = QPushButton("保存图片")
        self.play_video_btn = QPushButton("播放视频")
        self.save_video_btn = QPushButton("保存视频")
        
        self.single_image_btn.clicked.connect(self.load_single_image)
        self.multiple_images_btn.clicked.connect(self.load_multiple_images)
        self.camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.save_image_btn.clicked.connect(self.save_image)
        self.play_video_btn.clicked.connect(self.play_video)
        self.save_video_btn.clicked.connect(self.save_video)
        
        button_layout.addWidget(self.single_image_btn)
        button_layout.addWidget(self.multiple_images_btn)
        button_layout.addWidget(self.camera_btn)
        button_layout.addWidget(self.stop_camera_btn)
        button_layout.addWidget(self.save_image_btn)
        button_layout.addWidget(self.play_video_btn)
        button_layout.addWidget(self.save_video_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # 添加到主布局
        main_layout.addWidget(self.image_label, stretch=1)  # 图像区域可伸缩
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 初始状态
        self.save_image_btn.setEnabled(False)
        self.play_video_btn.setEnabled(False)
        self.save_video_btn.setEnabled(False)
        self.stop_camera_btn.setEnabled(False)
        
    def start_camera(self):
        """启动摄像头实时检测"""
        if self.camera_thread and self.camera_thread.running:
            QMessageBox.warning(self, "警告", "摄像头已经在运行中")
            return
            
        self.camera_thread = CameraThread(self.processor, self.display_frame)
        self.camera_thread.start()
        
        self.status_label.setText("摄像头实时检测中...")
        self.camera_btn.setEnabled(False)
        self.stop_camera_btn.setEnabled(True)
        
    def stop_camera(self):
        """停止摄像头检测"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.join()
            
        self.status_label.setText("摄像头已关闭")
        self.camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        
    def display_frame(self, frame):
        """显示摄像头捕获的帧 - 修改后的版本"""
        # 将OpenCV图像转换为Qt图像
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        
        # 创建QPixmap并保持宽高比缩放
        pixmap = QPixmap.fromImage(q_img)
        
        # 获取标签的当前大小（固定大小）
        label_size = self.image_label.size()
        
        # 计算保持宽高比的缩放
        scaled_pixmap = pixmap.scaled(
            label_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # 居中显示
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
        
    def load_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png)")
        
        if file_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            QApplication.processEvents()  # 更新UI
            
            processed_img, message = self.processor.process_single_image(file_path)
            self.progress_bar.setValue(50)
            
            if processed_img is not None:
                self.current_image = processed_img
                self.display_image(processed_img)
                self.save_image_btn.setEnabled(True)
                QMessageBox.information(self, "成功", message)
            else:
                QMessageBox.warning(self, "错误", message)
                
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)
            
    def load_multiple_images(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择包含图片的文件夹")
            
        if folder_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            QApplication.processEvents()  # 更新UI
            
            # 创建临时输出目录
            output_dir = os.path.join(os.path.dirname(folder_path), "temp_arm_pose_output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 处理图片并生成视频
            output_video = os.path.join(output_dir, "output_video.mp4")
            success, message = self.processor.process_images_to_video(folder_path, output_video, fps=24)
            
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)
            
            if success:
                self.video_path = output_video
                self.play_video_btn.setEnabled(True)
                self.save_video_btn.setEnabled(True)
                QMessageBox.information(self, "成功", message)
            else:
                QMessageBox.warning(self, "错误", message)
                
    def display_image(self, image):
        """显示单张图片 - 修改后的版本"""
        # 将OpenCV图像转换为Qt图像
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        
        # 创建QPixmap并保持宽高比缩放
        pixmap = QPixmap.fromImage(q_img)
        
        # 获取标签的当前大小（固定大小）
        label_size = self.image_label.size()
        
        # 计算保持宽高比的缩放
        scaled_pixmap = pixmap.scaled(
            label_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # 居中显示
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
        
    def save_image(self):
        if self.current_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图片", "", "JPEG 图片 (*.jpg);;PNG 图片 (*.png)")
                
            if file_path:
                cv2.imwrite(file_path, self.current_image)
                QMessageBox.information(self, "成功", "图片保存成功!")
                
    def save_video(self):
        if self.video_path:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存视频", "", "MP4 视频 (*.mp4)")
                
            if file_path:
                try:
                    import shutil
                    shutil.copy(self.video_path, file_path)
                    QMessageBox.information(self, "成功", "视频保存成功!")
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"保存视频失败: {str(e)}")
                    
    def play_video(self):
        if self.video_path:
            self.video_player = VideoPlayer()
            self.video_player.load_video(self.video_path)
            self.video_player.show()
            
    def closeEvent(self, event):
        # 停止摄像头线程
        if self.camera_thread and self.camera_thread.running:
            self.camera_thread.stop()
            self.camera_thread.join()
            
        # 关闭视频播放器
        if hasattr(self, 'video_player') and self.video_player:
            self.video_player.close()
            
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ArmPoseApp()
    window.show()
    sys.exit(app.exec_())