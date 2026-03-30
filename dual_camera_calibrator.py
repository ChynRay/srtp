"""
双相机独立标定管理类
支持相机 1 和相机 2 分别进行手眼标定，且机械臂运动范围不同
"""

import numpy as np
import cv2
import time
import os
import threading
import argparse
from utils.realsense_test import RealSenseCamera
from utils.piper_arm import robot_arm
from calibration_cal import CameraCalibration


class DualCameraCalibrator:
    """双相机独立标定管理器"""
    
    def __init__(self):
        self.camera1 = None  # 相机 1 (333422302278)
        self.camera2 = None  # 相机 2 (243222074585)
        self.robot = None
        self.calibration_in_progress = False
        self.lock = threading.Lock()
        
        # 相机序列号
        self.SERIAL_CAM1 = "333422302278"
        self.SERIAL_CAM2 = "243222074585"
        
        # 标定参数
        self.chessboard_size = (9, 6)
        self.square_size = 0.024  # 24mm
        
        # ===== 相机 1 关节运动范围（主相机，大范围运动）=====
        self.joint_ranges_cam1 = {
            'j1': [-60, -30, 0, 30, 60],      # 关节 1:5 个位置
            'j2': [0, 6, 12],                  # 关节 2:3 个位置
            'j3': [0, -10, -20, -30],         # 关节 3:4 个位置
            'j4': None,                        # 动态获取
            'j5': [0],                         # 关节 5:1 个位置
            'j6': [-75, -90, -105]            # 关节 6:3 个位置
        }
        
        # ===== 相机 2 关节运动范围（辅助相机，小范围运动）=====
        self.joint_ranges_cam2 = {
            'j1': [-10, 0, 10],               # 关节 1:3 个位置（范围更小）
            'j2': [0, 6, 12],                   # 关节 2:3 个位置（偏移）
            'j3': [0, -5, -10],               # 关节 3:3 个位置（范围更小）
            'j4': None,                        # 动态获取
            'j5': [10, 0, -10],                 # 关节 5:3 个位置（增加运动）
            'j6': [-15, -0, 15]             # 关节 6:3 个位置（范围更集中）
        }
    
    def init_cameras(self):
        """初始化两个相机"""
        try:
            self.camera1 = RealSenseCamera(self.SERIAL_CAM1)
            self.camera2 = RealSenseCamera(self.SERIAL_CAM2)
            print("[标定器] 相机初始化成功")
            return True
        except Exception as e:
            print(f"[标定器] 相机初始化失败：{e}")
            return False
    
    def init_robot(self):
        """初始化机械臂"""
        try:
            self.robot = robot_arm()
            print("[标定器] 机械臂初始化成功")
            return True
        except Exception as e:
            print(f"[标定器] 机械臂初始化失败：{e}")
            return False
    
    def get_joint_ranges(self, cam_id):
        """
        根据相机 ID 获取对应的关节运动范围
        
        :param cam_id: 相机 ID (1 或 2)
        :return: 关节范围字典
        """
        if cam_id == 1:
            return self.joint_ranges_cam1.copy()
        else:
            return self.joint_ranges_cam2.copy()
    
    def calibrate_camera(self, cam_id, cal_root, callback=None):
        """
        执行单个相机标定
        
        :param cam_id: 相机 ID (1 或 2)
        :param cal_root: 标定结果保存路径
        :param callback: 状态回调函数 callback(message)
        :return: 标定结果字典
        """
        with self.lock:
            if self.calibration_in_progress:
                if callback:
                    callback("标定正在进行中，请等待完成")
                return None
            self.calibration_in_progress = True
        
        try:
            # 选择相机
            if cam_id == 1:
                camera = self.camera1
                cam_name = "相机 1"
                serial = self.SERIAL_CAM1
                joint_ranges = self.joint_ranges_cam1
            else:
                camera = self.camera2
                cam_name = "相机 2"
                serial = self.SERIAL_CAM2
                joint_ranges = self.joint_ranges_cam2
            
            if callback:
                callback(f"{cam_name} 标定启动...")
            
            # 创建标定目录
            os.makedirs(cal_root, exist_ok=True)
            os.makedirs(os.path.join(cal_root, 'res'), exist_ok=True)
            os.makedirs(os.path.join(cal_root, 'real_trac'), exist_ok=True)
            
            # 初始化机械臂
            if not self.robot:
                self.init_robot()
            self.robot.enable_arm(True)
            
            if callback:
                callback(f"{cam_name} 机械臂使能成功")
            
            # 回到基准位置
            base_position = [55.848, 0.493, 242.414, 99.023, -2.186, 91.624]
            self.robot.move_arm_points(base_position)
            time.sleep(3)
            
            if callback:
                callback(f"{cam_name} 回到基准位置")
            
            # 获取当前关节角度（保持 j4 不变）
            original_joint = self.robot.piper.GetArmJointMsgs().joint_state
            j4 = original_joint.joint_4 / 1000
            joint_ranges['j4'] = j4
            
            # 启动相机
            camera.start_work()
            
            if callback:
                callback(f"{cam_name} 开始采集图像...")
            
            # 采集标定图像（使用对应相机的关节范围）
            self._capture_images(camera, cal_root, cam_name, joint_ranges, callback)
            
            if callback:
                callback(f"{cam_name} 图像采集完成，开始计算...")
            
            # 执行标定计算
            calibrator = CameraCalibration(
                cal_root,
                chessboard_size=self.chessboard_size,
                square_size=self.square_size
            )
            save_corner_root = os.path.join(cal_root, 'corner_points', 'detect_points.txt')
            calibrator.calibrate_work(save_corner_root)
            
            # 保存结果
            result = {
                "intrinsics_matrix": calibrator.intrinsics_matrix,
                "dist_coeffs": calibrator.dist_coeffs,
                "cam2base_H": calibrator.cam2base_H,
                "cam_name": cam_name,
                "serial": serial,
                "cal_root": cal_root
            }
            
            if callback:
                callback(f"{cam_name} 标定完成！")
            
            return result
            
        except Exception as e:
            error_msg = f"{cam_name} 标定失败：{str(e)}"
            print(f"[标定错误] {error_msg}")
            if callback:
                callback(error_msg)
            return None
            
        finally:
            # 清理
            try:
                self.robot.error_process()
                camera.stop_work() if hasattr(camera, 'stop_work') else None
            except:
                pass
            with self.lock:
                self.calibration_in_progress = False
    
    def _capture_images(self, camera, save_root, cam_name, joint_ranges, callback=None):
        """
        采集标定用图像
        
        :param camera: RealSenseCamera 实例
        :param save_root: 图像保存路径
        :param cam_name: 相机名称
        :param joint_ranges: 关节运动范围字典
        :param callback: 状态回调
        """
        res_dir = os.path.join(save_root, 'res')
        trac_dir = os.path.join(save_root, 'real_trac')
        
        img_count = 0
        real_trac = []
        
        # 遍历关节组合
        for j1 in joint_ranges['j1']:
            for j2 in joint_ranges['j2']:
                for j3 in joint_ranges['j3']:
                    for j5 in joint_ranges['j5']:
                        for j6 in joint_ranges['j6']:
                            newjoint = [j1, j2, j3, joint_ranges['j4'], j5, j6]
                            
                            # 限制关节范围
                            newjoint = self._constrain_joints(newjoint)
                            
                            try:
                                self.robot.move_arm_joints(newjoint)
                                time.sleep(1)
                                
                                # 获取末端位姿
                                endpose = self.robot.piper.GetArmEndPoseMsgs().end_pose
                                x, y, z = endpose.X_axis, endpose.Y_axis, endpose.Z_axis
                                rx, ry, rz = endpose.RX_axis, endpose.RY_axis, endpose.RZ_axis
                                real_trac.append([x, y, z, rx, ry, rz])
                                
                                # 保存图像
                                save_path = os.path.join(
                                    res_dir,
                                    f'{cam_name}_{img_count:03d}.png'
                                )
                                camera.work_flow(save_path)
                                img_count += 1
                                
                                if callback and img_count % 5 == 0:
                                    callback(f"{cam_name} 已采集 {img_count} 张图像...")
                                    
                            except Exception as e:
                                print(f"[图像采集错误] {e}")
                                continue
        
        # 保存轨迹数据
        real_trac = np.array(real_trac, dtype=np.float32) / 1000
        trac_file = os.path.join(trac_dir, 'trac.txt')
        np.savetxt(trac_file, real_trac, delimiter=',')
        
        print(f"[{cam_name}] 采集完成，共 {img_count} 张图像，轨迹已保存")
        if callback:
            callback(f"{cam_name} 已采集 {img_count} 张图像")
    
    def _constrain_joints(self, joints):
        """限制关节角度范围"""
        min_joints = [-150, 0.0, -175, -102, -75, -120]
        max_joints = [150, 180, 0, 102, 75, 120]
        
        constrained = []
        for j, minj, maxj in zip(joints, min_joints, max_joints):
            j = max(j, minj)
            j = min(j, maxj)
            constrained.append(j)
        return constrained
    
    def get_calibration_status(self):
        """获取标定状态"""
        with self.lock:
            return self.calibration_in_progress
    
    def stop_calibration(self):
        """停止标定（紧急）"""
        with self.lock:
            self.calibration_in_progress = False
        try:
            if self.robot:
                self.robot.error_process()
        except:
            pass
    
    def set_joint_ranges(self, cam_id, joint_ranges):
        """
        设置指定相机的关节运动范围
        
        :param cam_id: 相机 ID (1 或 2)
        :param joint_ranges: 关节范围字典
        """
        if cam_id == 1:
            self.joint_ranges_cam1 = joint_ranges
            print(f"[标定器] 相机 1 关节范围已更新")
        else:
            self.joint_ranges_cam2 = joint_ranges
            print(f"[标定器] 相机 2 关节范围已更新")
    
    def get_joint_ranges_info(self, cam_id):
        """
        获取指定相机的关节运动范围信息
        
        :param cam_id: 相机 ID (1 或 2)
        :return: 运动范围描述字符串
        """
        ranges = self.get_joint_ranges(cam_id)
        total_positions = (
            len(ranges['j1']) * len(ranges['j2']) * len(ranges['j3']) * 
            len(ranges['j5']) * len(ranges['j6'])
        )
        
        info = f"相机{cam_id} 标定运动范围:\n"
        info += f"  J1: {ranges['j1']} ({len(ranges['j1'])}个位置)\n"
        info += f"  J2: {ranges['j2']} ({len(ranges['j2'])}个位置)\n"
        info += f"  J3: {ranges['j3']} ({len(ranges['j3'])}个位置)\n"
        info += f"  J4: {ranges['j4']} (固定)\n"
        info += f"  J5: {ranges['j5']} ({len(ranges['j5'])}个位置)\n"
        info += f"  J6: {ranges['j6']} ({len(ranges['j6'])}个位置)\n"
        info += f"  总采集点数：{total_positions}"
        
        return info


# 全局标定器实例
global_calibrator = DualCameraCalibrator()


def get_calibrator():
    """获取全局标定器实例"""
    return global_calibrator

def default_callback(message):
    """默认回调函数，打印状态信息"""
    print(f"[状态] {message}")


def run_calibration(cam_id, cal_root):
    """
    执行标定流程
    
    :param cam_id: 相机 ID (1 或 2)
    :param cal_root: 标定结果保存路径
    :return: 标定结果
    """
    calibrator = get_calibrator()
    
    # 初始化相机
    if not calibrator.init_cameras():
        print("[错误] 相机初始化失败")
        return None
    
    # 执行标定
    result = calibrator.calibrate_camera(
        cam_id=cam_id,
        cal_root=cal_root,
        callback=default_callback
    )
    
    return result


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="双相机标定工具")
    parser.add_argument(
        "--cam_id", 
        type=int, 
        default=1, 
        choices=[1, 2],
        help="相机 ID (1 或 2)，默认：1"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./calibration_results",
        help="标定结果保存路径，默认：./calibration_results"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="仅显示关节运动范围信息，不执行标定"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("双相机标定工具")
    print("=" * 50)
    
    # 显示关节运动范围信息
    calibrator = get_calibrator()
    print(calibrator.get_joint_ranges_info(args.cam_id))
    print("=" * 50)
    
    if args.info:
        print("仅显示信息模式，退出")
        exit(0)
    
    # 执行标定
    try:
        result = run_calibration(
            cam_id=args.cam_id,
            cal_root=args.output
        )
        
        if result:
            print("\n" + "=" * 50)
            print("标定完成！")
            print(f"相机：{result['cam_name']}")
            print(f"序列号：{result['serial']}")
            print(f"保存路径：{result['cal_root']}")
            print(f"内参矩阵:\n{result['intrinsics_matrix']}")
            print(f"畸变系数：{result['dist_coeffs']}")
            print(f"手眼矩阵:\n{result['cam2base_H']}")
            print("=" * 50)
        else:
            print("\n[错误] 标定失败")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n[中断] 用户取消标定")
        calibrator.stop_calibration()
        exit(0)
    except Exception as e:
        print(f"\n[错误] 标定异常：{e}")
        calibrator.stop_calibration()
        exit(1)