import os
import cv2
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QSlider, QComboBox, 
                             QCheckBox, QListWidget, QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from ultralytics import YOLO
import numpy as np

class ArmKeypointDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = YOLO("robot_arm_pose_final.pt")  # 加载训练好的模型
        self.current_frame = None
        self.video_capture = None
        self.output_video_writer = None
        self.is_playing = False
        
        # 图片处理相关属性
        self.image_files = []  # 存储所有导入的图片路径
        self.current_image_index = -1  # 当前显示的图片索引
        self.processed_images = {}  # 存储处理后的图片 {文件路径: 处理后的图像}
        
        # 视频处理相关属性
        self.output_frames = []  # 存储处理后的视频帧
        self.video_processing = False
        
        self.init_ui()
        
    def init_ui(self):
        # 主窗口设置
        self.setWindowTitle('机械臂关键点检测系统 (支持多图和视频)')
        self.setGeometry(100, 100, 1400, 900)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        self.image_tab = QWidget()
        self.video_tab = QWidget()
        
        # 初始化图片和视频标签页
        self.init_image_tab()
        self.init_video_tab()
        
        self.tab_widget.addTab(self.image_tab, "图片处理")
        self.tab_widget.addTab(self.video_tab, "视频处理")
        
        # 右侧显示区域
        display_panel = QWidget()
        display_layout = QVBoxLayout(display_panel)
        
        self.original_label = QLabel('原始图像')
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 2px solid gray;")
        self.original_label.setMinimumSize(640, 360)
        
        self.result_label = QLabel('检测结果')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("border: 2px solid gray;")
        self.result_label.setMinimumSize(640, 360)
        
        # 导航按钮 (用于图片和视频)
        self.nav_panel = QWidget()
        nav_layout = QHBoxLayout(self.nav_panel)
        self.btn_prev = QPushButton('上一张/帧')
        self.btn_prev.clicked.connect(self.show_previous)
        self.btn_prev.setEnabled(False)
        self.btn_next = QPushButton('下一张/帧')
        self.btn_next.clicked.connect(self.show_next)
        self.btn_next.setEnabled(False)
        self.position_label = QLabel('0/0')
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.position_label)
        nav_layout.addWidget(self.btn_next)
        
        display_layout.addWidget(QLabel('原始图像:'))
        display_layout.addWidget(self.original_label)
        display_layout.addWidget(QLabel('检测结果:'))
        display_layout.addWidget(self.result_label)
        display_layout.addWidget(self.nav_panel)
        
        # 主布局
        main_layout.addWidget(self.tab_widget, stretch=1)
        main_layout.addWidget(display_panel, stretch=3)
        
        # 视频定时器
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('准备就绪')
    
    def init_image_tab(self):
        """初始化图片处理标签页"""
        layout = QVBoxLayout(self.image_tab)
        
        # 文件操作区域
        self.btn_load_image = QPushButton('添加图片')
        self.btn_load_image.clicked.connect(self.load_images)
        
        self.btn_load_images = QPushButton('批量添加图片')
        self.btn_load_images.clicked.connect(self.load_multiple_images)
        
        self.btn_clear_images = QPushButton('清空图片列表')
        self.btn_clear_images.clicked.connect(self.clear_image_list)
        
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(150)
        self.image_list.itemClicked.connect(self.display_selected_image)
        
        # 处理按钮
        self.btn_process_image = QPushButton('处理当前图片')
        self.btn_process_image.clicked.connect(self.process_current_image)
        self.btn_process_image.setEnabled(False)
        
        self.btn_process_all_images = QPushButton('处理所有图片')
        self.btn_process_all_images.clicked.connect(self.process_all_images)
        self.btn_process_all_images.setEnabled(False)
        
        # 保存按钮
        self.btn_save_image = QPushButton('保存当前结果')
        self.btn_save_image.clicked.connect(self.save_current_image_result)
        self.btn_save_image.setEnabled(False)
        
        self.btn_save_all_images = QPushButton('保存所有结果')
        self.btn_save_all_images.clicked.connect(self.save_all_image_results)
        self.btn_save_all_images.setEnabled(False)
        
        # 添加到布局
        layout.addWidget(QLabel("图片列表 (最多50张):"))
        layout.addWidget(self.image_list)
        layout.addWidget(self.btn_load_image)
        layout.addWidget(self.btn_load_images)
        layout.addWidget(self.btn_clear_images)
        layout.addSpacing(10)
        layout.addWidget(self.btn_process_image)
        layout.addWidget(self.btn_process_all_images)
        layout.addSpacing(10)
        layout.addWidget(self.btn_save_image)
        layout.addWidget(self.btn_save_all_images)
        layout.addStretch()
    
    def init_video_tab(self):
        """初始化视频处理标签页"""
        layout = QVBoxLayout(self.video_tab)
        
        # 文件操作区域
        self.btn_load_video = QPushButton('导入视频')
        self.btn_load_video.clicked.connect(self.load_video)
        
        # 处理按钮
        self.btn_process_video_frame = QPushButton('处理当前帧')
        self.btn_process_video_frame.clicked.connect(self.process_video_frame)
        self.btn_process_video_frame.setEnabled(False)
        
        self.btn_process_all_video = QPushButton('处理整个视频')
        self.btn_process_all_video.clicked.connect(self.process_all_video_frames)
        self.btn_process_all_video.setEnabled(False)
        
        # 播放控制
        self.btn_play_video = QPushButton('播放')
        self.btn_play_video.clicked.connect(self.toggle_video_playback)
        self.btn_play_video.setEnabled(False)
        
        # 保存按钮
        self.btn_save_video = QPushButton('保存处理后的视频')
        self.btn_save_video.clicked.connect(self.save_video_result)
        self.btn_save_video.setEnabled(False)
        
        # 参数设置区域
        self.kpt_check = QCheckBox('显示关键点')
        self.kpt_check.setChecked(True)
        
        self.line_check = QCheckBox('显示连接线')
        self.line_check.setChecked(True)
        
        self.conf_label = QLabel('置信度阈值: 0.5')
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        
        # 添加到布局
        layout.addWidget(self.btn_load_video)
        layout.addSpacing(10)
        layout.addWidget(self.btn_process_video_frame)
        layout.addWidget(self.btn_process_all_video)
        layout.addSpacing(10)
        layout.addWidget(self.btn_play_video)
        layout.addWidget(self.btn_save_video)
        layout.addSpacing(20)
        layout.addWidget(self.kpt_check)
        layout.addWidget(self.line_check)
        layout.addWidget(QLabel('置信度阈值:'))
        layout.addWidget(self.conf_slider)
        layout.addWidget(self.conf_label)
        layout.addStretch()
    
    # ================== 图片处理相关方法 ==================
    def load_images(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png)")
        if file_path:
            self.add_image_to_list(file_path)
    
    def load_multiple_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多张图片", "", "图片文件 (*.jpg *.jpeg *.png)")
        if file_paths:
            for file_path in file_paths:
                self.add_image_to_list(file_path)
    
    def add_image_to_list(self, file_path):
        if len(self.image_files) >= 50:
            QMessageBox.warning(self, "警告", "已达到最大图片数量限制(50张)")
            return
        
        if file_path not in self.image_files:
            self.image_files.append(file_path)
            self.image_list.addItem(os.path.basename(file_path))
            self.update_image_controls()
            
            # 如果是第一张图片，自动显示
            if len(self.image_files) == 1:
                self.current_image_index = 0
                self.display_image_at_index(0)
    
    def clear_image_list(self):
        self.image_files.clear()
        self.image_list.clear()
        self.processed_images.clear()
        self.current_image_index = -1
        self.original_label.clear()
        self.result_label.clear()
        self.position_label.setText('0/0')
        self.update_image_controls()
        self.status_bar.showMessage('已清空图片列表')
    
    def display_selected_image(self, item):
        index = self.image_list.row(item)
        self.display_image_at_index(index)
    
    def display_image_at_index(self, index):
        if 0 <= index < len(self.image_files):
            self.current_image_index = index
            file_path = self.image_files[index]
            
            # 显示原始图像
            self.current_frame = cv2.imread(file_path)
            self.show_image(self.current_frame, self.original_label)
            
            # 如果已有处理结果，显示结果
            if file_path in self.processed_images:
                self.show_image(self.processed_images[file_path], self.result_label)
                self.btn_save_image.setEnabled(True)
            else:
                self.result_label.clear()
                self.btn_save_image.setEnabled(False)
            
            # 更新位置标签
            self.position_label.setText(f'{index + 1}/{len(self.image_files)}')
            self.update_navigation_buttons()
    
    def process_current_image(self):
        if self.current_image_index >= 0 and self.current_frame is not None:
            file_path = self.image_files[self.current_image_index]
            # self.process_image(file_path, self.current_frame)
            self.process_frame(file_path, self.current_frame)
    
    def process_all_images(self):
        if not self.image_files:
            return
        
        self.btn_process_all_images.setEnabled(False)
        self.status_bar.showMessage('正在处理所有图片...')
        
        for i, file_path in enumerate(self.image_files):
            if file_path not in self.processed_images:  # 只处理未处理的图片
                image = cv2.imread(file_path)
                # self.process_image(file_path, image)
                self.process_frame(file_path, image)  # 使用正确的方法名
                QApplication.processEvents()  # 更新UI
            
            # 更新进度
            self.status_bar.showMessage(f'正在处理图片: {i + 1}/{len(self.image_files)}')
        
        self.btn_process_all_images.setEnabled(True)
        self.status_bar.showMessage('所有图片处理完成')
        
        # 显示当前图片的处理结果
        if self.current_image_index >= 0:
            file_path = self.image_files[self.current_image_index]
            if file_path in self.processed_images:
                self.show_image(self.processed_images[file_path], self.result_label)
    
    def save_current_image_result(self):
        if self.current_image_index >= 0:
            file_path = self.image_files[self.current_image_index]
            if file_path in self.processed_images:
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "保存结果图片", 
                    os.path.splitext(file_path)[0] + "_result.jpg",
                    "JPEG图片 (*.jpg);;PNG图片 (*.png)")
                
                if save_path:
                    result_image = self.processed_images[file_path]
                    cv2.imwrite(save_path, result_image)
                    self.status_bar.showMessage(f'结果已保存到: {save_path}')
    
    def save_all_image_results(self):
        if not self.processed_images:
            QMessageBox.warning(self, "警告", "没有可保存的处理结果")
            return
        
        folder = QFileDialog.getExistingDirectory(self, "选择保存结果的文件夹")
        if folder:
            success_count = 0
            for file_path, result_image in self.processed_images.items():
                try:
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    save_path = os.path.join(folder, f"{base_name}_result.jpg")
                    cv2.imwrite(save_path, result_image)
                    success_count += 1
                except Exception as e:
                    print(f"保存失败 {file_path}: {str(e)}")
            
            self.status_bar.showMessage(f'成功保存 {success_count}/{len(self.processed_images)} 个结果')
            QMessageBox.information(self, "完成", f"已保存 {success_count} 个结果图片到:\n{folder}")
    
    # ================== 视频处理相关方法 ==================
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame
                self.show_image(frame, self.original_label)
                self.position_label.setText(f'1/{self.total_frames}')
                
                # 启用相关按钮
                self.btn_process_video_frame.setEnabled(True)
                self.btn_process_all_video.setEnabled(True)
                self.btn_play_video.setEnabled(True)
                
                self.status_bar.showMessage(f'已加载视频: {os.path.basename(file_path)}')
                self.output_frames = []  # 清空之前的输出帧
                self.update_navigation_buttons()
    
    def process_video_frame(self):
        if self.current_frame is not None and self.video_capture is not None:
            current_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self.process_frame(self.current_frame, current_pos)
    
    def process_all_video_frames(self):
        if self.video_capture is None:
            return
        
        self.btn_process_all_video.setEnabled(False)
        self.btn_load_video.setEnabled(False)
        self.status_bar.showMessage('正在处理视频...')
        
        # 获取视频参数
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 重置视频到第一帧
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.output_frames = []
        
        # 获取处理参数
        conf = self.conf_slider.value() / 100
        show_kpt = self.kpt_check.isChecked()
        show_line = self.line_check.isChecked()
        
        # 处理每一帧
        frame_count = 0
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            # 处理当前帧
            results = self.model.predict(
                frame,
                conf=conf,
                verbose=False
            )
            
            # 绘制结果
            annotated_frame = results[0].plot(
                kpt_line=show_line,
                kpt_radius=5 if show_kpt else 0
            )
            
            self.output_frames.append(annotated_frame)
            frame_count += 1
            self.position_label.setText(f'{frame_count}/{self.total_frames}')
            QApplication.processEvents()  # 更新UI
            
            # 显示处理进度
            self.status_bar.showMessage(f'正在处理视频: {frame_count}/{self.total_frames}')
        
        self.btn_process_all_video.setEnabled(True)
        self.btn_load_video.setEnabled(True)
        self.btn_save_video.setEnabled(True)
        self.status_bar.showMessage('视频处理完成')
        
        # 显示第一帧结果
        if self.output_frames:
            self.show_image(self.output_frames[0], self.result_label)
    
    def toggle_video_playback(self):
        if self.is_playing:
            self.video_timer.stop()
            self.btn_play_video.setText('播放')
            self.is_playing = False
        else:
            self.video_timer.start(1000 // 30)  # 30 FPS
            self.btn_play_video.setText('暂停')
            self.is_playing = True
    
    def update_video_frame(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame
                self.show_image(frame, self.original_label)
                
                # 更新位置
                current_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                self.position_label.setText(f'{current_pos}/{self.total_frames}')
                
                # 如果有处理结果，显示结果
                if self.output_frames and current_pos - 1 < len(self.output_frames):
                    self.show_image(self.output_frames[current_pos - 1], self.result_label)
            else:
                # 视频结束，回到第一帧
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.toggle_video_playback()
    
    def save_video_result(self):
        if not self.output_frames:
            QMessageBox.warning(self, "警告", "没有可保存的视频处理结果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存视频", "", "MP4视频 (*.mp4)")
        
        if file_path:
            # 获取视频参数
            fps = self.video_capture.get(cv2.CAP_PROP_FPS) if self.video_capture else 30
            width = self.output_frames[0].shape[1]
            height = self.output_frames[0].shape[0]
            
            # 创建VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            
            # 写入所有帧
            for frame in self.output_frames:
                out.write(frame)
            
            out.release()
            self.status_bar.showMessage(f'视频已保存到: {file_path}')
    
    # ================== 通用方法 ==================
    def show_previous(self):
        if self.tab_widget.currentIndex() == 0:  # 图片标签页
            if self.current_image_index > 0:
                self.display_image_at_index(self.current_image_index - 1)
        else:  # 视频标签页
            if self.video_capture is not None:
                current_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if current_pos > 0:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                    ret, frame = self.video_capture.read()
                    if ret:
                        self.current_frame = frame
                        self.show_image(frame, self.original_label)
                        self.position_label.setText(f'{current_pos}/{self.total_frames}')
                        
                        # 如果有处理结果，显示结果
                        if self.output_frames and current_pos - 1 < len(self.output_frames):
                            self.show_image(self.output_frames[current_pos - 1], self.result_label)
    
    def show_next(self):
        if self.tab_widget.currentIndex() == 0:  # 图片标签页
            if self.current_image_index < len(self.image_files) - 1:
                self.display_image_at_index(self.current_image_index + 1)
        else:  # 视频标签页
            if self.video_capture is not None:
                current_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if current_pos < self.total_frames - 1:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_pos + 1)
                    ret, frame = self.video_capture.read()
                    if ret:
                        self.current_frame = frame
                        self.show_image(frame, self.original_label)
                        self.position_label.setText(f'{current_pos + 2}/{self.total_frames}')
                        
                        # 如果有处理结果，显示结果
                        if self.output_frames and current_pos + 1 < len(self.output_frames):
                            self.show_image(self.output_frames[current_pos + 1], self.result_label)
    
    def update_navigation_buttons(self):
        if self.tab_widget.currentIndex() == 0:  # 图片标签页
            self.btn_prev.setEnabled(self.current_image_index > 0)
            self.btn_next.setEnabled(self.current_image_index < len(self.image_files) - 1)
        else:  # 视频标签页
            if self.video_capture is not None:
                current_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self.btn_prev.setEnabled(current_pos > 0)
                self.btn_next.setEnabled(current_pos < self.total_frames - 1)
            else:
                self.btn_prev.setEnabled(False)
                self.btn_next.setEnabled(False)
    
    def process_frame(self, frame, frame_index=None):
        try:
            # 获取当前参数设置
            conf = self.conf_slider.value() / 100
            show_kpt = self.kpt_check.isChecked()
            show_line = self.line_check.isChecked()
            
            # 处理当前帧
            results = self.model.predict(
                frame,
                conf=conf,
                verbose=False
            )
            
            # 绘制结果
            annotated_frame = results[0].plot(
                kpt_line=show_line,
                kpt_radius=5 if show_kpt else 0
            )
            
            # 显示结果
            self.show_image(annotated_frame, self.result_label)
            
            # 根据当前标签页存储结果
            if self.tab_widget.currentIndex() == 0:  # 图片标签页
                file_path = self.image_files[self.current_image_index]
                self.processed_images[file_path] = annotated_frame
                self.btn_save_image.setEnabled(True)
            else:  # 视频标签页
                if frame_index is not None and frame_index < len(self.output_frames):
                    self.output_frames[frame_index] = annotated_frame
                else:
                    self.output_frames.append(annotated_frame)
                self.btn_save_video.setEnabled(True)
            
            self.status_bar.showMessage('处理完成')
            return True
        except Exception as e:
            self.status_bar.showMessage(f'处理错误: {str(e)}')
            return False
    
    def update_image_controls(self):
        has_images = len(self.image_files) > 0
        self.btn_process_image.setEnabled(has_images)
        self.btn_process_all_images.setEnabled(has_images)
        self.btn_save_image.setEnabled(has_images and self.current_image_index >= 0 and 
                                     self.image_files[self.current_image_index] in self.processed_images)
        self.btn_save_all_images.setEnabled(len(self.processed_images) > 0)
        self.update_navigation_buttons()
    
    def update_conf_label(self, value):
        self.conf_label.setText(f'置信度阈值: {value/100:.2f}')
    
    def show_image(self, image, label):
        """将OpenCV图像显示在QLabel上"""
        if image is not None:
            # 转换颜色空间 BGR -> RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def closeEvent(self, event):
        if self.video_capture is not None:
            self.video_capture.release()
        if hasattr(self, 'output_video_writer') and self.output_video_writer is not None:
            self.output_video_writer.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用现代UI风格
    window = ArmKeypointDetector()
    window.show()
    sys.exit(app.exec_())