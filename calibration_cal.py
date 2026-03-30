import glob
import pickle
import numpy as np
import cv2
import csv
import os 
from utils.handeye_calibration import handeye_calibration        #眼在手外
#from utils.handeye_calibration import eye_in_hand_calibration   眼在手上

class CameraCalibration():
    def __init__(self,calibration_root,chessboard_size=[9, 6], square_size=0.024):
        """
        手眼标定
        :param calibration_root: 保存路径
        :param chessboard_size: 棋盘格尺寸 默认为[9, 6]
        :param square_size: 棋盘格每个方格的尺寸 (单位:m)

        function:
        calibrate_work() 进行手眼标定
        """
        self.root = calibration_root
        self.img_root= os.path.join(self.root,'res')
        self.trac_root= os.path.join(self.root,'real_trac')
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        self.save_root=os.path.join(self.root,'vis')

        self.intrinsics_matrix = None
        self.dist_coeffs = None
        self.cam2base_H = None
    
    def generate_checkboard(self):
        """
        生成棋盘格理论点坐标
        :return: 棋盘格理论点坐标
        """
        chessboard_size = self.chessboard_size
        square_size = self.square_size
        object_point = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        object_point[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        object_point *= square_size
        self.object_point = object_point
        return object_point
    
    def find_checkboard(self,save_corner_points_root,vis=True):
        # 读取轨迹
        flag=[]
        # self.read_log_to_position()
        self.read_real_position()
        img_file=sorted(glob.glob(os.path.join(self.img_root,"*.png")))
        self.object_points=[]
        self.corner_points=[]
        os.makedirs(os.path.join(*save_corner_points_root.split('/')[:-1]),exist_ok=True)
        with open(save_corner_points_root, "w") as f:
            for img_path in img_file[:]:
                f.write(img_path+'\n')
                img = cv2.imread(img_path)
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
                flag.append(ret)
                if ret == True:
                    # 亚像素级精确化角点位置
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 11, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                if ret:
                    self.object_points.append(self.object_point)
                    self.corner_points.append(corners)
                if vis:
                    os.makedirs(self.save_root,exist_ok=True)
                    if ret == True:
                        cv2.drawChessboardCorners(img, self.chessboard_size, corners, ret)
                        cv2.imwrite(os.path.join(self.save_root,os.path.basename(img_path)),img)
                        for j in corners:
                            f.write('{} {}\n'.format(j[0][0],j[0][1]))
        self.positions=self.positions[:] 
        self.positions=self.positions[flag] 
        self.img_shape=gray.shape[::-1] # 获取图像的尺寸 （width x height）

    def intrinsics_calibration(self,intrinsics_matrix=None,dist_coeffs=None):
        if intrinsics_matrix is None:
            self.object_points=np.array(self.object_points)
            self.corner_points=np.array(self.corner_points)
            ret, self.intrinsics_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, self.corner_points, self.img_shape, None, None)
        else:
            self.intrinsics_matrix = intrinsics_matrix
            self.dist_coeffs = dist_coeffs

            
    def hand_eye_calibration(self):
        # 将positions数组中的前3列数据除以1000
        self.positions[:,:3]=self.positions[:,:3]/1000
        # 将positions数组赋值给end2base_xyzrxryrz
        end2base_xyzrxryrz=self.positions
        # 创建一个空列表，用于存储board2cam_xyzrxryrz
        board2cam_xyzrxryrz_list=[]
        # 遍历object_points和corner_points
        for obj, leftp in zip(self.object_points,self.corner_points):

            # 调用solvePnP_board2cam函数，计算board2cam_xyzrxryrz
            xyzrxryrz=self.solvePnP_board2cam(obj,leftp,self.intrinsics_matrix,self.dist_coeffs)
            # 将计算结果添加到board2cam_xyzrxryrz_list列表中
            board2cam_xyzrxryrz_list.append(xyzrxryrz)

        # 将board2cam_xyzrxryrz_list列表转换为numpy数组
        board2cam_xyzrxryrz_list=np.array(board2cam_xyzrxryrz_list)
        # 调用handeye_calibration函数，计算cam2base_H
        cam2base_H= handeye_calibration(end2base_xyzrxryrz,board2cam_xyzrxryrz_list)
        # 将计算结果赋值给self.cam2base_H
        self.cam2base_H=cam2base_H
        # 返回cam2base_H
        return cam2base_H
    def save_calibration_data(self, root=None):
        if root is None:
            root = self.root
        # 保存标定数据
        with open(os.path.join(root, "calibration_data.pickle"), "wb") as f:
            pickle.dump({
                "intrinsics_matrix": self.intrinsics_matrix,
                "dist_coeffs": self.dist_coeffs,
                "cam2base_H":self.cam2base_H
            }, f)
        with open(os.path.join(root, "calibration_data.txt"), "w") as f:
            f.write("intrinsics_matrix:\n")
            f.write(str(self.intrinsics_matrix) + "\n")
            f.write("dist_coeffs:\n")
            f.write(str(self.dist_coeffs) + "\n")
            f.write("cam2base_H:\n")
            f.write(str(self.cam2base_H) + "\n")

    def load_calibration_data(self,root):
        # 加载标定数据
        with open(os.path.join(root, "calibration_data.pickle"), "rb") as f:
            data = pickle.load(f)
            self.intrinsics_matrix = data["intrinsics_matrix"]
            self.dist_coeffs = data["dist_coeffs"]
            self.cam2base_H = data["cam2base_H"]
    def solvePnP_board2cam(self,object_points, image_points, camera_matrix, distortion_coefficients):
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coefficients)
        xyzrxryrz = np.concatenate((tvec, rvec), axis=0).reshape((1, 6)).tolist()[0]
        return xyzrxryrz
    def show_arg(self):
        print("intrinsics_matrix:\n")
        print(self.intrinsics_matrix)
        print("dist_coeffs:\n")
        print(self.dist_coeffs)
        print("cam2base_H:\n")
        print(self.cam2base_H)
    def read_log_to_position(self):
        trac_name = os.listdir(self.trac_root)[0]
        with open(os.path.join(self.trac_root,trac_name), 'r') as f:
            reader = csv.reader(f,delimiter=',')
            self.positions = [row for row in reader]
    
        self.positions=[i[3:9] for i in self.positions[2:]]
        self.positions=np.array(self.positions,dtype=np.float32)
    def read_real_position(self):
        trac_name = os.listdir(self.trac_root)[0]
        with open(os.path.join(self.trac_root,trac_name), 'r') as f:
            reader = csv.reader(f,delimiter=',')
            self.positions = [row for row in reader]
    
        self.positions=[i for i in self.positions]
        self.positions=np.array(self.positions,dtype=np.float32)
    def calibrate_work(self,save_corner_points_root):
        self.generate_checkboard()
        self.find_checkboard(save_corner_points_root)
        self.intrinsics_calibration()
        self.hand_eye_calibration()
        self.save_calibration_data(self.root)
        self.show_arg()

def main(root,chessboard_size,square_size,):
    camera_cal=CameraCalibration(root,chessboard_size,square_size)
    save_corner_points_root=os.path.join(root,'corner_points','detect_points.txt')
    camera_cal.calibrate_work(save_corner_points_root)



if __name__ == '__main__':
    root="calibrationeyetohand_res/20250719 cal v11.0"
    chessboard_size=(9,6)
    square_size=24*1e-3
    main(root,chessboard_size,square_size)