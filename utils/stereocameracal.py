import pickle
import cv2
import numpy as np
import glob
import os
from pathlib import Path


class SteraoCamera():  # 双目相机标定类
    def __init__(self):
        pass

    def stereo_camera_calibration(self,root,save_path,camera_name=['left','right'],chessboard_size=(9, 6),square_size=0.0121):
        #从单目相机标定转换为双目相机标定
        for camera in camera_name:
            Path(os.path.join(save_path,camera)).mkdir(parents=True, exist_ok=True)

        # 用于存储标定数据
        object_points = []  # 3D点
        left_image_points = []  # 左图像的2D点
        right_image_points = []  # 右图像的2D点

        # 3D 世界坐标系中的点（假设棋盘格位于z=0平面）
        object_point = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        object_point[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        object_point *= square_size

        # 获取左侧和右侧图像的文件路径
        left_images = sorted(glob.glob(os.path.join(root,'left',"*.tif")))
        right_images = sorted(glob.glob(os.path.join(root,'right',"*.tif")))

        # 检查左侧和右侧图像数量是否相等
        assert len(left_images) == len(right_images), "左右图像数量不匹配！"

        # 遍历图像并提取角点
        for left_img_path, right_img_path in zip(left_images, right_images):
            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)

            # 转换为灰度图像
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            # 查找棋盘格角点
            ret_left, corners_left = cv2.findChessboardCorners(left_gray, chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(right_gray, chessboard_size, None)

            # 可视化角点结果
            if ret_left and ret_right:
                cv2.drawChessboardCorners(left_img, chessboard_size, corners_left, ret_left)
                cv2.drawChessboardCorners(right_img, chessboard_size, corners_right, ret_right)
                #保存中间结果
                cv2.imwrite(os.path.join(save_path, 'left', os.path.basename(left_img_path)), left_img)
                cv2.imwrite(os.path.join(save_path, 'right', os.path.basename(right_img_path)), right_img)

            if ret_left and ret_right:
                object_points.append(object_point) 
                # 亚像素级精确化角点位置
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left_subpix = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
                corners_right_subpix = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)
                left_image_points.append(corners_left_subpix)
                right_image_points.append(corners_right_subpix)

        # 首先进行单目相机标定
        print("开始左相机标定...")
        ret_left, camera_matrix_left, dist_coeffs_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            object_points, left_image_points, left_gray.shape[::-1], None, None)

        print("开始右相机标定...")
        ret_right, camera_matrix_right, dist_coeffs_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            object_points, right_image_points, right_gray.shape[::-1], None, None)

        # 使用单目标定的结果进行双目标定
        print("开始双目相机标定...")
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F = cv2.stereoCalibrate(
            object_points, left_image_points, right_image_points,
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            left_gray.shape[::-1],
            criteria=criteria_stereo,
            flags=cv2.CALIB_FIX_INTRINSIC  # 使用单目标定的内参
        )

        # 计算重投影误差来衡量标定的精度
        total_error = 0
        for i in range(len(object_points)):
            img_points_left2, _ = cv2.projectPoints(object_points[i], rvecs_left[i], tvecs_left[i], camera_matrix_left, dist_coeffs_left)
            error_left = cv2.norm(left_image_points[i], img_points_left2, cv2.NORM_L2) / len(img_points_left2)
            
            img_points_right2, _ = cv2.projectPoints(object_points[i], rvecs_right[i], tvecs_right[i], camera_matrix_right, dist_coeffs_right)
            error_right = cv2.norm(right_image_points[i], img_points_right2, cv2.NORM_L2) / len(img_points_right2)
            
            total_error += (error_left + error_right) / 2

        mean_error = total_error / len(object_points)
        print(f"重投影误差: {mean_error}")


        # 保存标定结果
        # np.savez(os.path.join(save_path, 'stereo_calibration.npz'), 
        #         camera_matrix_left=camera_matrix_left,
        #         dist_coeffs_left=dist_coeffs_left,
        #         camera_matrix_right=camera_matrix_right,
        #         dist_coeffs_right=dist_coeffs_right,
        #         # R=R, T=T, E=E, F=F)

        self.camera_matrix_left = camera_matrix_left
        self.dist_coeffs_left = dist_coeffs_left
        self.camera_matrix_right = camera_matrix_right
        self.dist_coeffs_right = dist_coeffs_right
        self.R = R
        self.T = T
        self.E = E  
        self.F = F
        self.object_points = object_points
        self.image_points_left = left_image_points
        self.image_points_right = right_image_points
        
        self.save_calibration_data(save_path)


    def show_calibration_result(self):  
        if self.camera_matrix_left is None or self.camera_matrix_right is None:
            print("请先进行标定！")
            return

        # 显示标定结果
        print("\n标定结果：")
        print("标定图片数量{}".format(len(self.object_points)))

        print("左相机矩阵：\n", self.camera_matrix_left)
        print("左相机畸变系数：\n", self.dist_coeffs_left)
        print("右相机矩阵：\n", self.camera_matrix_right)
        print("右相机畸变系数：\n", self.dist_coeffs_right)
        print("旋转矩阵：\n", self.R)
        print("平移向量：\n", self.T)
        print("本质矩阵：\n", self.E)
        print("基础矩阵：\n", self.F)

    def load_calibration_data(self,root):
        # 加载标定数据
        with open(os.path.join(root, "calibration_data.pickle"), "rb") as f:
            data = pickle.load(f)
            self.camera_matrix_left = data["camera_matrix_left"]
            self.dist_coeffs_left = data["dist_coeffs_left"]
            self.camera_matrix_right = data["camera_matrix_right"]
            self.dist_coeffs_right = data["dist_coeffs_right"]
            self.R = data["R"]
            self.T = data["T"]
            self.E = data["E"]
            self.F = data["F"]
            self.object_points = data["object_points"]
            self.image_points_left = data["image_points_left"]

    def save_calibration_data(self, root):
        # 保存标定数据
        with open(os.path.join(root, "calibration_data.pickle"), "wb") as f:
            pickle.dump({
                "camera_matrix_left": self.camera_matrix_left,
                "dist_coeffs_left": self.dist_coeffs_left,
                "camera_matrix_right": self.camera_matrix_right,
                "dist_coeffs_right": self.dist_coeffs_right,
                "R": self.R,
                "T": self.T,
                "E": self.E,
                "F": self.F,
                "object_points": self.object_points,
                "image_points_left": self.image_points_left,
                "image_points_right": self.image_points_right
            }, f)
    
    def get_calibration_res(self):
        return self.camera_matrix_left, self.dist_coeffs_left, self.camera_matrix_right, self.dist_coeffs_right, self.R, self.T


if __name__ == "__main__":
    root="20250402 calibration data"
    saveroot="20250402 calibration data/res"