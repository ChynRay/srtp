# test_final_merged.py
import sys
import threading
import cv2
import numpy as np
import pyrealsense2 as rs
import pickle
import transforms3d as tfs
import time
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFrame

import calutils1123
from ikcal1 import posetoangle
from utils.piper_arm import robot_arm
from dual_camera_calibrator import get_calibrator

# ==================== 共享资源 ====================
target_position = None
exit_flag = False
click_points = []
depth_values = []
lock = threading.Lock()

color_frame_cache1 = None
depth_frame_cache1 = None
color_frame_cache2 = None
depth_frame_cache2 = None

# ==================== 相机配置 ====================
TARGET_SERIAL1 = "333422302278"
TARGET_SERIAL2 = "243222074585"

# ==================== 机械臂控制类 ====================
class Transform:
    def __init__(self):
        self.pixel_coords = None
        self.robot = robot_arm()
        '初始化相机参数'
        with open('calibration_results/calibration_data.pickle', 'rb') as f:
            data = pickle.load(f)
        self.camera_intrinsics_matrix = data['intrinsics_matrix']
        self.camera_dist_coeffs = data['dist_coeffs']
        self.cam2base_H = data['cam2base_H']
               
    def set_extrinsics(self):
        cam2base_H = self.cam2base_H     
        t, R, scale, shear = tfs.affines.decompose(cam2base_H)
        return t, R
    
    def image_to_camera(self, pixel_coords, depth_value):
        z = depth_value
        camera_intrinsics_matrix_inv = np.linalg.inv(self.camera_intrinsics_matrix)
        camera_coords = z * camera_intrinsics_matrix_inv @ pixel_coords
        return camera_coords
    
    def camera_to_base(self, camera_coords):
        t, R = self.set_extrinsics()
        base_coords = np.dot(R, camera_coords) + t
        return base_coords
    
    def robot_move(self, target_pose):
        self.robot.enable_arm(True)
        time.sleep(1)
        self.robot.move_arm_joints(target_pose)
        return True
    
    def robot_disable(self):
        self.robot.enable_arm(False)
        print("机械臂已禁用")
        return True

# ==================== 相机采集线程 ====================
def camera1_task():
    '''配置相机 1，获取图像'''
    global exit_flag, pipeline1, align1, color_frame_cache1, depth_frame_cache1
    
    pipeline1 = rs.pipeline()
    config1 = rs.config()
    config1.enable_device(TARGET_SERIAL1)
    config1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    config1.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    pipeline1.start(config1)
    align1 = rs.align(rs.stream.color)
    
    while not exit_flag:
        frames = pipeline1.wait_for_frames(5000)
        aligned_frames = align1.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
        
        with lock:
            color_frame_cache1 = np.asanyarray(color_frame.get_data())
            depth_frame_cache1 = depth_frame
            
    pipeline1.stop()
    print("相机 1 线程已结束")

def camera2_task():
    '''配置相机 2，获取图像'''
    global exit_flag, pipeline2, align2, color_frame_cache2, depth_frame_cache2
    
    pipeline2 = rs.pipeline()
    config2 = rs.config()
    config2.enable_device(TARGET_SERIAL2)
    config2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    config2.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    pipeline2.start(config2)
    align2 = rs.align(rs.stream.color)
    
    while not exit_flag:
        frames = pipeline2.wait_for_frames(5000)
        aligned_frames = align2.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
        
        with lock:
            color_frame_cache2 = np.asanyarray(color_frame.get_data())
            depth_frame_cache2 = depth_frame
            
    pipeline2.stop()
    print("相机 2 线程已结束")

# ==================== 机械臂控制线程 ====================
def robot_task():
    """机械臂线程：持续监听目标位置，有新目标就移动"""
    global exit_flag, target_position
    
    tran = Transform()
    while not exit_flag:
        current_target = None
        current_angle = None
        
        if target_position is not None:
            current_target = target_position
            current_angle = posetoangle(current_target)
            target_position = None

        if current_angle is not None:
            print(f"机械臂移动到：{current_target}")
            tran.robot_move(current_angle)
            current_target = None

        time.sleep(0.1)
        
    tran.robot_disable()
    print("机械臂线程已结束")

# ==================== PyQt5 信号通信类 ====================
class SignalEmitter(QObject):
    """用于线程间信号通信"""
    status_update = pyqtSignal(str)
    calibration_done = pyqtSignal(dict)
    btn_enable = pyqtSignal(int)

# ==================== PyQt5 主界面 ====================
class RobotControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("机械臂视觉控制系统（双摄像头）")
        self.setGeometry(200, 200, 1200, 900)
        
        self.tran = Transform()
        
        # 标定器（不初始化相机，避免与相机线程冲突）
        self.calibrator = get_calibrator()
        # self.calibrator.init_cameras()  # ← 注释掉，由相机线程管理
        # self.calibrator.init_robot()    # ← 注释掉，标定时单独初始化
        
        # 标定线程
        self.calibration_thread1 = None
        self.calibration_thread2 = None
        
        # 信号发射器（在主线程创建，修复 Qt 线程错误）
        self.signal_emitter = SignalEmitter()
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(30)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # ===== 左侧：双摄像头画面（垂直排列）=====
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        
        # 摄像头 1 显示
        self.video_label1 = QLabel("摄像头 1 画面")
        self.video_label1.setFixedSize(800, 450)
        self.video_label1.setStyleSheet("border: 2px solid #00ff00; background-color: #2a2a2a; color: white;")
        self.video_label1.mousePressEvent = self.on_mouse_click
        left_layout.addWidget(self.video_label1)
        
        # 摄像头 1 标签
        self.cam1_name = QLabel("相机 1 (主相机 - 用于选点)")
        self.cam1_name.setStyleSheet("color: #00ff00; font-weight: bold;")
        self.cam1_name.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.cam1_name)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #666666;")
        separator.setFixedHeight(2)
        left_layout.addWidget(separator)
        
        # 摄像头 2 显示
        self.video_label2 = QLabel("摄像头 2 画面")
        self.video_label2.setFixedSize(800, 450)
        self.video_label2.setStyleSheet("border: 2px solid #0088ff; background-color: #2a2a2a; color: white;")
        left_layout.addWidget(self.video_label2)
        
        # 摄像头 2 标签
        self.cam2_name = QLabel("相机 2 (辅助相机)")
        self.cam2_name.setStyleSheet("color: #0088ff; font-weight: bold;")
        self.cam2_name.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.cam2_name)
        
        main_layout.addWidget(left_panel)
        
        # ===== 右侧：按钮面板 =====
        panel = QWidget()
        panel.setFixedWidth(300)
        btn_layout = QVBoxLayout(panel)
        
        # 标定按钮（新增）
        self.btn_calibrate_cam1 = QPushButton("📷 相机 1 标定")
        self.btn_calibrate_cam1.setStyleSheet("background-color: #00aa00; color: white; font-weight: bold;")
        
        self.btn_calibrate_cam2 = QPushButton("📷 相机 2 标定")
        self.btn_calibrate_cam2.setStyleSheet("background-color: #0066cc; color: white; font-weight: bold;")
        
        # 原有按钮
        self.btn_home = QPushButton("回零位")
        self.btn_grip_open = QPushButton("打开夹爪")
        self.btn_grip_close = QPushButton("关闭夹爪")
        self.btn_move_to_point = QPushButton("移动到选点")
        self.btn_clear_points = QPushButton("清除选点")
        self.btn_stop = QPushButton("🛑 紧急停止")
        self.btn_stop.setStyleSheet("background-color: #cc0000; color: white; font-weight: bold;")
        
        for btn in [self.btn_calibrate_cam1, self.btn_calibrate_cam2,
                    self.btn_home, self.btn_grip_open, 
                    self.btn_grip_close, self.btn_move_to_point, 
                    self.btn_clear_points, self.btn_stop]:
            btn.setMinimumHeight(45)
            btn_layout.addWidget(btn)
        
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        btn_layout.addWidget(self.status_label)
        
        self.points_label = QLabel("已选点：0/3")
        self.points_label.setStyleSheet("color: yellow;")
        btn_layout.addWidget(self.points_label)
        
        main_layout.addWidget(panel)
        
        # ===== 连接按钮信号 =====
        self.btn_calibrate_cam1.clicked.connect(self.on_calibrate_cam1)
        self.btn_calibrate_cam2.clicked.connect(self.on_calibrate_cam2)
        self.btn_home.clicked.connect(self.on_home)
        self.btn_grip_open.clicked.connect(lambda: self.on_grip(True))
        self.btn_grip_close.clicked.connect(lambda: self.on_grip(False))
        self.btn_move_to_point.clicked.connect(self.on_move_to_point)
        self.btn_clear_points.clicked.connect(self.on_clear_points)
        self.btn_stop.clicked.connect(self.on_emergency_stop)
        
        # ===== 连接自定义信号（在主线程）=====
        self.signal_emitter.status_update.connect(self.update_status)
        self.signal_emitter.btn_enable.connect(self.on_btn_enable)
        
    def on_btn_enable(self, cam_id):
        """在主线程启用标定按钮"""
        if cam_id == 1:
            self.btn_calibrate_cam1.setEnabled(True)
        else:
            self.btn_calibrate_cam2.setEnabled(True)
        
    def on_calibrate_cam1(self):
        """启动相机 1 标定线程"""
        if self.calibrator.get_calibration_status():
            self.update_status("标定正在进行中，请等待完成")
            return
        
        self.btn_calibrate_cam1.setEnabled(False)
        self.calibration_thread1 = threading.Thread(
            target=self._run_calibration,
            args=(1, "相机 1")
        )
        self.calibration_thread1.start()
    
    def on_calibrate_cam2(self):
        """启动相机 2 标定线程"""
        if self.calibrator.get_calibration_status():
            self.update_status("标定正在进行中，请等待完成")
            return
        
        self.btn_calibrate_cam2.setEnabled(False)
        self.calibration_thread2 = threading.Thread(
            target=self._run_calibration,
            args=(2, "相机 2")
        )
        self.calibration_thread2.start()
    
    def _run_calibration(self, cam_id, cam_name):
        """执行标定（后台线程）"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cal_root = f"calibrationeyetohand_res/{cam_name}_{timestamp}"
        
        def status_callback(msg):
            # 通过信号发送到主线程
            self.signal_emitter.status_update.emit(msg)

        result = self.calibrator.calibrate_camera(
            cam_id=cam_id,
            cal_root=cal_root,
            callback=status_callback
        )
        
        # 通过信号在主线程恢复按钮状态
        self.signal_emitter.btn_enable.emit(cam_id)
        
        if result:
            self.signal_emitter.status_update.emit(f"{cam_name} 标定完成，结果已保存")
    
    def update_display(self):
        """定时更新显示画面"""
        global color_frame_cache1, color_frame_cache2, click_points
        
        # ===== 更新摄像头 1 画面 =====
        with lock:
            color_img1 = color_frame_cache1
        
        if color_img1 is not None:
            display_img1 = color_img1.copy()
            
            for i, (px, py) in enumerate(click_points):
                color = (0, 255, 0)
                cv2.circle(display_img1, (px, py), 8, color, -1)
                cv2.putText(display_img1, str(i+1), (px+10, py+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            rgb1 = cv2.cvtColor(display_img1, cv2.COLOR_BGR2RGB)
            h1, w1, ch1 = rgb1.shape
            bytes_per_line1 = ch1 * w1
            qt_img1 = QImage(rgb1.data, w1, h1, bytes_per_line1, QImage.Format_RGB888)
            self.video_label1.setPixmap(QPixmap.fromImage(qt_img1).scaled(
                self.video_label1.width(), self.video_label1.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        
        # ===== 更新摄像头 2 画面 =====
        with lock:
            color_img2 = color_frame_cache2
        
        if color_img2 is not None:
            display_img2 = color_img2.copy()
            rgb2 = cv2.cvtColor(display_img2, cv2.COLOR_BGR2RGB)
            h2, w2, ch2 = rgb2.shape
            bytes_per_line2 = ch2 * w2
            qt_img2 = QImage(rgb2.data, w2, h2, bytes_per_line2, QImage.Format_RGB888)
            self.video_label2.setPixmap(QPixmap.fromImage(qt_img2).scaled(
                self.video_label2.width(), self.video_label2.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        
        self.points_label.setText(f"已选点：{len(click_points)}/3")
        
    def on_mouse_click(self, event):
        """处理鼠标点击事件（仅在摄像头 1 上生效）"""
        global click_points, depth_values, depth_frame_cache1
        
        if event.source() != self.video_label1:
            return
            
        if depth_frame_cache1 is None:
            return
            
        x = event.x()
        y = event.y()
        
        pixmap = self.video_label1.pixmap()
        if pixmap:
            ratio_x = 1280 / self.video_label1.width()
            ratio_y = 720 / self.video_label1.height()
            img_x = int(x * ratio_x)
            img_y = int(y * ratio_y)
            
            with lock:
                click_points.append((img_x, img_y))
                depth = depth_frame_cache1.get_distance(img_x, img_y) if depth_frame_cache1 else 0
                depth_values.append(depth)
                
            # 通过实例信号发送（在主线程）
            self.signal_emitter.status_update.emit(f"已选点：({img_x}, {img_y}), 深度：{depth:.3f}m")
            
    def on_home(self):
        global target_position
        self.update_status("回零位...")
        with lock:
            target_position = [0, 0, 0, 0, 0, 0]
        self.update_status("已回零")
        
    def on_grip(self, open_state):
        self.tran.robot.enable_arm(True)
        self.update_status(f"夹爪{'打开' if open_state else '关闭'}")
        
    def on_move_to_point(self):
        global target_position, click_points, depth_values
        
        if len(click_points) < 3:
            self.update_status("请至少选择 3 个点！")
            return
            
        with lock:
            sample_point = click_points[-3:]
            depth_sample = depth_values[-3:]
            target_points = []
            
            for i in range(3):
                x, y = sample_point[i]
                depth = depth_sample[i]
                if depth > 0:
                    pixel_coords = np.array([x, y, 1])
                    camera_coords = self.tran.image_to_camera(pixel_coords, depth)
                    base_coords = self.tran.camera_to_base(camera_coords)
                    target_point = [1000*base_coords[0], 1000*base_coords[1], 1000*base_coords[2]]
                    target_points.append(target_point)
            
            if len(target_points) == 3:
                normal = calutils1123.get_normal(target_points[-1], target_points[-2], target_points[-3])
                z = normal
                y = [0, 1, 0]
                x = np.cross(y, z)
                euler = calutils1123.get_tfeuler(x, y, z)
                target_position = [target_points[-1][0], target_points[-1][1], 
                                  target_points[-1][2], euler[0], euler[1], euler[2]]
                self.update_status("目标位姿已设置，机械臂移动中...")
            else:
                self.update_status("深度数据无效，请重新选点")
                
    def on_clear_points(self):
        global click_points, depth_values
        with lock:
            click_points.clear()
            depth_values.clear()
        self.update_status("已清除选点")
        
    def on_emergency_stop(self):
        global exit_flag
        self.calibrator.stop_calibration()
        self.tran.robot_disable()
        self.update_status("🛑 紧急停止！")
        
    def update_status(self, message):
        self.status_label.setText(message)
        print(f"[状态] {message}")
        
    def closeEvent(self, event):
        global exit_flag
        exit_flag = True
        time.sleep(0.5)
        self.calibrator.stop_calibration()
        self.tran.robot_disable()
        event.accept()

# ==================== 主函数 ====================
if __name__ == "__main__":
    # ===== 启动后台线程 =====
    camera_thread1 = threading.Thread(target=camera1_task)
    camera_thread2 = threading.Thread(target=camera2_task)
    robot_thread = threading.Thread(target=robot_task)
    
    camera_thread1.start()
    camera_thread2.start()
    robot_thread.start()
    
    # ===== 启动 PyQt5 GUI（必须在创建任何 QObject 之前）=====
    app = QApplication(sys.argv)
    window = RobotControlWindow()
    window.show()
    
    # ===== 等待线程结束 =====
    camera_thread1.join()
    camera_thread2.join()
    robot_thread.join()
    
    print("程序已退出")