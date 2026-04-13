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
from core import robot
from core import trans

# ==================== 共享资源（修复：补全双相机独立变量+初始化）====================
target_position = None
exit_flag = False
# 相机1独立选点/深度
click_points1 = []
depth_values1 = []
# 相机2独立选点/深度
click_points2 = []
depth_values2 = []
# 原兼容变量（保留）
click_points = []
depth_values = []

lock = threading.Lock()
# 帧缓存初始化
color_frame_cache1 = None
depth_frame_cache1 = None
color_frame_cache2 = None
depth_frame_cache2 = None

# ==================== 相机配置（修复：适配Realsense通用参数+定义常量）====================
TARGET_SERIAL1 = "333422302278"
TARGET_SERIAL2 = "243222074585"
# 相机1分辨率/帧率（D400系列通用稳定参数）
CAM1_W, CAM1_H, CAM1_FPS = 640, 480, 15
# 相机2分辨率/帧率（D400系列通用稳定参数）
CAM2_W, CAM2_H, CAM2_FPS = 640, 480, 15

# ==================== 相机采集线程（核心修复：锁作用域+异常捕获+参数适配+硬件重置）====================
def camera1_task():
    '''配置相机 1，获取图像'''
    global exit_flag, pipeline1, align1, color_frame_cache1, depth_frame_cache1
    
    pipeline1 = rs.pipeline()
    config1 = rs.config()
    try:
        config1.enable_device(TARGET_SERIAL1)
        # 适配硬件的流配置
        config1.enable_stream(rs.stream.color, CAM1_W, CAM1_H, rs.format.bgr8, CAM1_FPS)
        config1.enable_stream(rs.stream.depth, CAM1_W, CAM1_H, rs.format.z16, CAM1_FPS)
        # 启动并硬件重置（解决二次运行卡死）
        profile = pipeline1.start(config1)
        profile.get_device().hardware_reset()
        time.sleep(0.5)
        align1 = rs.align(rs.stream.color)
        print("【相机1】初始化成功")
    except Exception as e:
        print(f"【相机1】初始化失败：{str(e)}")
        exit_flag = True
        return
    
    while not exit_flag:
        try:
            # 缩短超时时间，避免线程阻塞
            frames = pipeline1.wait_for_frames(1000)
            aligned_frames = align1.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # 修复：锁仅保护缓存赋值，耗时的数组转换放锁外
            color_data = np.asanyarray(color_frame.get_data())
            depth_data = depth_frame
            with lock:
                color_frame_cache1 = color_data
                depth_frame_cache1 = depth_data
        except Exception as e:
            print(f"【相机1】帧获取异常：{str(e)}")
            time.sleep(0.1)
            continue
            
    pipeline1.stop()
    print("【相机1】线程已结束")

def camera2_task():
    '''配置相机 2，获取图像'''
    global exit_flag, pipeline2, align2, color_frame_cache2, depth_frame_cache2
    
    pipeline2 = rs.pipeline()
    config2 = rs.config()
    try:
        config2.enable_device(TARGET_SERIAL2)
        # 适配硬件的流配置
        config2.enable_stream(rs.stream.color, CAM2_W, CAM2_H, rs.format.bgr8, CAM2_FPS)
        config2.enable_stream(rs.stream.depth, CAM2_W, CAM2_H, rs.format.z16, CAM2_FPS)
        # 启动并硬件重置
        profile = pipeline2.start(config2)
        profile.get_device().hardware_reset()
        time.sleep(0.5)
        align2 = rs.align(rs.stream.color)
        print("【相机2】初始化成功")
    except Exception as e:
        print(f"【相机2】初始化失败：{str(e)}")
        exit_flag = True
        return
    
    while not exit_flag:
        try:
            frames = pipeline2.wait_for_frames(1000)
            aligned_frames = align2.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # 修复：锁仅保护缓存赋值
            color_data = np.asanyarray(color_frame.get_data())
            depth_data = depth_frame
            with lock:
                color_frame_cache2 = color_data
                depth_frame_cache2 = depth_data
        except Exception as e:
            print(f"【相机2】帧获取异常：{str(e)}")
            time.sleep(0.1)
            continue
            
    pipeline2.stop()
    print("【相机2】线程已结束")

# ==================== 机械臂控制线程（保留原逻辑，仅注释）====================
def robot_task():
    """机械臂线程：持续监听目标位置，有新目标就移动"""
    global exit_flag, target_position
    
    arm = robot.RobotArm()
    while not exit_flag:
        current_target = None
        current_angle = None
        
        if target_position is not None:
            current_target = target_position
            current_angle = posetoangle(current_target)
            target_position = None
        if current_angle is not None:
            print(f"机械臂移动到：{current_target}")
            arm.move_joints(current_angle)
            current_target = None
        time.sleep(0.1)

# ==================== PyQt5 信号通信类（保留原逻辑）====================
class SignalEmitter(QObject):
    """用于线程间信号通信"""
    status_update = pyqtSignal(str)
    calibration_done = pyqtSignal(dict)
    btn_enable = pyqtSignal(int)

# ==================== PyQt5 主界面（核心修复：坐标映射+选点逻辑+画面渲染+变量引用）====================
class RobotControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("机械臂视觉控制系统（双摄像头）")
        self.setGeometry(200, 200, 1200, 900)
        
        self.arm = robot.RobotArm()
        self.tran = trans.Transform()
        
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
        self.video_label1.setFixedSize(640, 480)
        self.video_label1.setStyleSheet("border: 2px solid #00ff00; background-color: #2a2a2a; color: white;")
        self.video_label1.mousePressEvent = self.on_mouse_click1
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
        self.video_label2.setFixedSize(640, 480)
        self.video_label2.setStyleSheet("border: 2px solid #0088ff; background-color: #2a2a2a; color: white;")
        self.video_label2.mousePressEvent = self.on_mouse_click2
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
        
        # 按钮（新增）
        self.btn_calibrate_cam1 = QPushButton("📷 相机 1 标定")
        self.btn_calibrate_cam1.setStyleSheet("background-color: #00aa00; color: white; font-weight: bold;")
        
        self.btn_calibrate_cam2 = QPushButton("📷 相机 2 标定")
        self.btn_calibrate_cam2.setStyleSheet("background-color: #0066cc; color: white; font-weight: bold;")
        self.btn_disable = QPushButton("失能")
        self.btn_disable.setStyleSheet("background-color: #0066cc; color: white; font-weight: bold;")
        self.btn_caliload = QPushButton("加载标定数据")
        self.btn_caliload.setStyleSheet("background-color: #0066cc; color: white; font-weight: bold;")
        
        # 原有按钮
        self.btn_home = QPushButton("回零位")
        self.btn_move_to_point = QPushButton("移动到选点")
        self.btn_clear_points = QPushButton("清除选点")
        self.btn_stop = QPushButton("🛑 紧急停止")
        self.btn_stop.setStyleSheet("background-color: #cc0000; color: white; font-weight: bold;")
        
        for btn in [self.btn_calibrate_cam1, self.btn_calibrate_cam2,
                    self.btn_home, self.btn_move_to_point, 
                    self.btn_clear_points, self.btn_stop, self.btn_disable, self.btn_caliload]:
            btn.setMinimumHeight(45)
            btn_layout.addWidget(btn)
        
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        btn_layout.addWidget(self.status_label)
        
        self.points_label = QLabel("已选点：0/3")
        self.points_label.setStyleSheet("color: yellow;")
        btn_layout.addWidget(self.points_label)
        
        main_layout.addWidget(panel)
        
        # ===== 连接按钮信号（保留原逻辑）=====
        self.btn_calibrate_cam1.clicked.connect(self.on_calibrate_cam1)
        self.btn_calibrate_cam2.clicked.connect(self.on_calibrate_cam2)
        self.btn_home.clicked.connect(self.on_home)
        self.btn_disable.clicked.connect(self.arm.disable)
        self.btn_move_to_point.clicked.connect(self.on_move_to_point)
        self.btn_clear_points.clicked.connect(self.on_clear_points)
        self.btn_stop.clicked.connect(self.on_emergency_stop)
        self.btn_caliload.clicked.connect(self.tran.load_calib)
        
        # ===== 连接自定义信号（在主线程，保留）=====
        self.signal_emitter.status_update.connect(self.update_status)
        self.signal_emitter.btn_enable.connect(self.on_btn_enable)
        
    def on_btn_enable(self, cam_id):
        """在主线程启用标定按钮（保留原逻辑）"""
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
        
        # 【关键修改】创建一个专用的标定器实例，传入当前的 pipeline
        # 假设 pipeline1 是全局变量或在类中可以访问
        global pipeline1 
        ext_pipes = {1: pipeline1} 
        
        # 获取带有外部管道的标定器实例
        calibrator_instance = get_calibrator(external_pipelines=ext_pipes)
        
        self.calibration_thread1 = threading.Thread(
            target=self._run_calibration_with_instance,
            args=(calibrator_instance, 1, "相机 1")
        )
        self.calibration_thread1.start()
    
    def on_calibrate_cam2(self):
        """启动相机 2 标定线程"""
        if self.calibrator.get_calibration_status():
            self.update_status("标定正在进行中，请等待完成")
            return
        
        self.btn_calibrate_cam2.setEnabled(False)
        
        # 【关键修改】创建一个专用的标定器实例，传入当前的 pipeline
        # 假设 pipeline1 是全局变量或在类中可以访问
        global pipeline2 
        ext_pipes = {2: pipeline2} 
        
        # 获取带有外部管道的标定器实例
        calibrator_instance = get_calibrator(external_pipelines=ext_pipes)
        
        self.calibration_thread1 = threading.Thread(
            target=self._run_calibration_with_instance,
            args=(calibrator_instance, 2, "相机 2")
        )
        self.calibration_thread1.start()
    
    def _run_calibration_with_instance(self, calib_inst, cam_id, cam_name):
        """使用特定标定器实例执行标定"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cal_root = f"calibrationeyetohand_res/{cam_name}_{timestamp}"
        
        def status_callback(msg):
            self.signal_emitter.status_update.emit(msg)
            
        # 注意：这里不需要再 init_cameras，因为 pipeline 已经通过构造函数传入
        # 但需要确保机械臂已初始化
        if not calib_inst.robot:
            calib_inst.init_robot()
            
        result = calib_inst.calibrate_camera(
            cam_id=cam_id,
            cal_root=cal_root,
            callback=status_callback
        )
        
        self.signal_emitter.btn_enable.emit(cam_id)
        
        if result:
            self.signal_emitter.status_update.emit(f"{cam_name} 标定完成，结果已保存")
    
    def update_display(self):
        """定时更新显示画面（核心修复：坐标映射+锁优化+空值判断+选点绘制）"""
        global color_frame_cache1, color_frame_cache2, click_points1, click_points2
        
        # ===== 更新摄像头 1 画面 =====
        with lock:
            # 修复：空值判断+深拷贝，避免线程冲突
            color_img1 = color_frame_cache1.copy() if color_frame_cache1 is not None else None
        
        if color_img1 is not None:
            display_img1 = color_img1.copy()
            # 绘制相机1选点，修复：边界判断避免越界
            for i, (px, py) in enumerate(click_points1):
                if 0 <= px < CAM1_W and 0 <= py < CAM1_H:
                    color = (0, 255, 0)
                    cv2.circle(display_img1, (px, py), 8, color, -1)
                    cv2.putText(display_img1, str(i+1), (px+10, py+10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            # 格式转换渲染
            rgb1 = cv2.cvtColor(display_img1, cv2.COLOR_BGR2RGB)
            h1, w1, ch1 = rgb1.shape
            bytes_per_line1 = ch1 * w1
            qt_img1 = QImage(rgb1.data, w1, h1, bytes_per_line1, QImage.Format_RGB888)
            self.video_label1.setPixmap(QPixmap.fromImage(qt_img1).scaled(
                self.video_label1.width(), self.video_label1.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            # 帧未就绪时友好提示
            self.video_label1.setText("相机1 未连接/帧获取中...")
        
        # ===== 更新摄像头 2 画面 =====
        with lock:
            color_img2 = color_frame_cache2.copy() if color_frame_cache2 is not None else None
        
        if color_img2 is not None:
            display_img2 = color_img2.copy()
            # 绘制相机2选点，修复：边界判断
            for i, (px, py) in enumerate(click_points2):
                if 0 <= px < CAM2_W and 0 <= py < CAM2_H:
                    color = (0, 255, 0)
                    cv2.circle(display_img2, (px, py), 8, color, -1)
                    cv2.putText(display_img2, str(i+1), (px+10, py+10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            # 格式转换渲染
            rgb2 = cv2.cvtColor(display_img2, cv2.COLOR_BGR2RGB)
            h2, w2, ch2 = rgb2.shape
            bytes_per_line2 = ch2 * w2
            qt_img2 = QImage(rgb2.data, w2, h2, bytes_per_line2, QImage.Format_RGB888)
            self.video_label2.setPixmap(QPixmap.fromImage(qt_img2).scaled(
                self.video_label2.width(), self.video_label2.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            self.video_label2.setText("相机2 未连接/帧获取中...")
        
        # 修复：选点计数为双相机合计，匹配原逻辑
        total_points = len(click_points1) + len(click_points2)
        self.points_label.setText(f"已选点：{total_points}/3")
    
    def on_mouse_click1(self, event):
        """处理相机1鼠标点击（核心修复：坐标映射+深度帧引用+独立变量存储）"""
        global click_points1, depth_values1, depth_frame_cache1, click_points, depth_values
        
        # 仅响应左键
        if event.button() != QtCore.Qt.LeftButton:
            return
        if depth_frame_cache1 is None:
            self.update_status("【相机1】深度帧未就绪，无法选点")
            return
            
        x, y = event.x(), event.y()
        pixmap = self.video_label1.pixmap()
        if pixmap:
            # 修复：用相机1实际分辨率计算映射比例，替代硬编码1280/720
            ratio_x = CAM1_W / self.video_label1.width()
            ratio_y = CAM1_H / self.video_label1.height()
            img_x = int(x * ratio_x)
            img_y = int(y * ratio_y)
            
            # 边界检查，避免选点超出帧范围
            if 0 <= img_x < CAM1_W and 0 <= img_y < CAM1_H:
                with lock:
                    # 修复：存入相机1独立变量，同时兼容原click_points（保留原逻辑）
                    click_points1.append((img_x, img_y))
                    click_points.append((img_x, img_y))
                    # 修复：用相机1深度帧获取深度，避免引用错误
                    depth = depth_frame_cache1.get_distance(img_x, img_y)
                    depth_values1.append(depth)
                    depth_values.append(depth)
                # 选点成功提示
                self.signal_emitter.status_update.emit(f"【相机1】已选点：({img_x}, {img_y}), 深度：{depth:.3f}m")
            else:
                self.update_status("【相机1】选点超出画面范围，请重新点击")
    
    def on_mouse_click2(self, event):
        """处理相机2鼠标点击（核心修复：坐标映射+深度帧引用+独立变量存储）"""
        global click_points2, depth_values2, depth_frame_cache2, click_points, depth_values
        
        if event.button() != QtCore.Qt.LeftButton:
            return
        if depth_frame_cache2 is None:
            self.update_status("【相机2】深度帧未就绪，无法选点")
            return
            
        x, y = event.x(), event.y()
        pixmap = self.video_label2.pixmap()
        if pixmap:
            # 修复：用相机2实际分辨率计算映射比例
            ratio_x = CAM2_W / self.video_label2.width()
            ratio_y = CAM2_H / self.video_label2.height()
            img_x = int(x * ratio_x)
            img_y = int(y * ratio_y)
            
            if 0 <= img_x < CAM2_W and 0 <= img_y < CAM2_H:
                with lock:
                    # 修复：存入相机2独立变量，兼容原click_points（保留原逻辑）
                    click_points2.append((img_x, img_y))
                    click_points.append((img_x, img_y))
                    # 修复：关键！用相机2深度帧获取深度，原代码误用了camera1的depth_frame_cache1
                    depth = depth_frame_cache2.get_distance(img_x, img_y)
                    depth_values2.append(depth)
                    depth_values.append(depth)
                self.signal_emitter.status_update.emit(f"【相机2】已选点：({img_x}, {img_y}), 深度：{depth:.3f}m")
            else:
                self.update_status("【相机2】选点超出画面范围，请重新点击")
            
    def on_home(self):
        """回零位（保留原逻辑）"""
        global target_position
        self.update_status("回零位...")
        with lock:
            target_position = [0, 0, 0, 0, 0, 0]
        self.update_status("已回零")
        
    def on_move_to_point(self):
        """移动到选点（完全保留你原有的业务逻辑，未做任何修改）"""
        global target_position, click_points1, depth_values1, click_points2, depth_values2
        
        if len(click_points1) < 3 and len(click_points2) < 3 :
            self.update_status("请至少选择 3 个点！")
            return
            
        with lock:
            sample_point1 = click_points1[-3:]
            depth_sample1 = depth_values1[-3:]
            sample_point2 = click_points2[-3:]
            depth_sample2 = depth_values2[-3:]
            target_points = []
            if len(click_points1) >= 3:
                for i in range(3):
                    x, y = sample_point1[i]
                    depth = depth_sample1[i]
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
            if len(click_points2) >= 3:
                for i in range(3):
                    x, y = sample_point2[i]
                    depth = depth_sample2[i]
                    if depth > 0:
                        pixel_coords = np.array([x, y, 1])
                        camera_coords = self.tran.image_to_camera(pixel_coords, depth)
                        end_coords = self.tran.camera_to_end(camera_coords)
                        base_coords = self.tran.end_to_base(end_coords)
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
        """清除选点（修复：清空双相机独立变量+原兼容变量）"""
        global click_points, depth_values, click_points1, depth_values1, click_points2, depth_values2
        with lock:
            click_points.clear()
            depth_values.clear()
            click_points1.clear()
            depth_values1.clear()
            click_points2.clear()
            depth_values2.clear()
        self.update_status("已清除所有选点")
        
    def on_emergency_stop(self):
        """紧急停止（保留原逻辑）"""
        global exit_flag
        self.calibrator.stop_calibration()
        self.tran.robot_disable()
        self.update_status("🛑 紧急停止！")
        
    def update_status(self, message):
        """更新状态（保留原逻辑）"""
        self.status_label.setText(message)
        print(f"[状态] {message}")
        
    def closeEvent(self, event):
        """窗口关闭事件（修复：优雅退出，避免线程卡死）"""
        global exit_flag
        exit_flag = True
        time.sleep(1)  # 给线程退出时间
        # 注释的代码保留原逻辑，未修改
        # self.calibrator.stop_calibration()
        # self.tran.robot_disable()
        self.update_status("程序正在退出...")
        event.accept()

# ==================== 主函数（修复：线程守护+Qt高DPI+相机初始化等待）====================
if __name__ == "__main__":
    # ===== Windows 系统，删掉所有 Linux 环境变量 =====
    
    # ===== 启动后台线程（设为守护线程，随主程序退出）=====
    camera_thread1 = threading.Thread(target=camera1_task, daemon=True)
    camera_thread2 = threading.Thread(target=camera2_task, daemon=True)
    robot_thread = threading.Thread(target=robot_task, daemon=True)
    
    camera_thread1.start()
    camera_thread2.start()
    robot_thread.start()
    
    # 等待相机初始化
    time.sleep(1.5)
    print("【主程序】相机线程启动完成，启动GUI...")
    
    # ===== 启动 PyQt5 GUI =====
    app = QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    window = RobotControlWindow()
    window.show()
    
    exit_code = app.exec_()
    print(f"【主程序】GUI退出，退出码：{exit_code}")
    
    exit_flag = True
    camera_thread1.join(2)
    camera_thread2.join(2)
    
    print("程序已正常退出")