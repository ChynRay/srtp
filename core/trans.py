import numpy as np
import pickle
import utils.piper_arm 
import transforms3d as tfs
import calutils1123

class Transform():
    def __init__(self, path, is_eye_in_hand=False):
        self.intrinsic = None
        self.robot = utils.piper_arm.robot_arm()
        self.path = path
        self.is_eye_in_hand = is_eye_in_hand # 标记是否为眼在手上
        self.cam2end_H = None # 眼在手上：相机到末端的变换
        self.cam2base_H = None # 眼在外：相机到基座的变换

    def load_calib(self):
        with open(self.path, 'rb') as f:
            data = pickle.load(f)
        self.camera_intrinsics_matrix = data['intrinsics_matrix']
        self.camera_dist_coeffs = data['dist_coeffs']
        
        # 根据配置加载不同的矩阵
        if self.is_eye_in_hand:
            # 眼在手上：加载相机到末端的变换矩阵
            # 注意：请确保你的 pickle 文件里有这个键，或者 cam2base_H 其实存的就是手眼矩阵
            if 'cam2end_H' in data:
                self.cam2end_H = data['cam2end_H']
            else:
                # 兼容旧数据：假设 cam2base_H 里存的就是手眼矩阵
                print("警告：未找到 cam2end_H，尝试使用 cam2base_H 作为手眼矩阵")
                self.cam2end_H = data['cam2base_H']
        else:
            # 眼在外：加载相机到基座的变换矩阵
            self.cam2base_H = data['cam2base_H']
        print('加载标定数据成功！')

    def set_extrinsics(self):
        """仅用于眼在外模式"""
        if self.cam2base_H is None:
            raise Exception("未加载眼在外标定数据")
        t, R, scale, shear = tfs.affines.decompose(self.cam2base_H)
        return t, R
    
    def set_hand_eye_extrinsics(self):
        """仅用于眼在手上模式"""
        if self.cam2end_H is None:
            raise Exception("未加载眼在手上标定数据")
        t, R, scale, shear = tfs.affines.decompose(self.cam2end_H)
        return t, R

    def image_to_camera(self, pixel_coords, depth_value):
        z = depth_value
        camera_intrinsics_matrix_inv = np.linalg.inv(self.camera_intrinsics_matrix)
        camera_coords = z * camera_intrinsics_matrix_inv @ pixel_coords
        return camera_coords
    
    def camera_to_base(self, camera_coords):
        """眼在外模式：相机 -> 基座"""
        if self.is_eye_in_hand:
            raise Exception("当前为眼在手上模式，请使用 camera_to_end + end_to_base")
        t, R = self.set_extrinsics()
        base_coords = np.dot(R, camera_coords) + t
        return base_coords
    
    def camera_to_end(self, camera_coords):
        """眼在手上模式：相机 -> 末端"""
        if not self.is_eye_in_hand:
            raise Exception("当前为眼在外模式，请使用 camera_to_base")
        
        t, R = self.set_hand_eye_extrinsics()
        end_coords = np.dot(R, camera_coords) + t
        return end_coords
    
    def end_to_base(self, end_coords):
        """通用：末端 -> 基座 (依赖机械臂实时位姿)"""
        pose = self.robot.read_point_position()
        # pose: [x, y, z, rx, ry, rz] (mm, deg)
        
        # 1. 提取平移向量 t (m)
        t = np.array([pose[0], pose[1], pose[2]])/1000.0
        
        # 2. 提取旋转矩阵 R
        # 注意：确保 get_eulertf 接收的是角度(deg)并内部转换为弧度
        R = calutils1123.get_eulertf(pose[3], pose[4], pose[5])
        
        # 3. 变换: P_base = R * P_end + t
        base_coords = np.dot(R, np.asarray(end_coords).reshape(3,)) + t
        
        return base_coords