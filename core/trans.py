import numpy as np
import pickle
import utils.piper_arm 
import transforms3d as tfs
import calutils1123

class Transform():
    def __init__(self,path):
        self.intrinsic = None
        self.cam2base = None
        self.robot = utils.piper_arm.robot_arm()
        self.path = path
    
    def load_calib(self):
        with open(self.path, 'rb') as f:
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
    
    def camera_to_end(self,camera_coords):
        t, R = self.set_extrinsics()
        end_coords = np.dot(R,camera_coords) + t
        return end_coords
    
    def end_to_base(self, end_coords):
        pose = self.robot.read_point_position()
        pose = [i/1000 for i in pose]
        t = np.array(pose)
        R = calutils1123.get_eulertf(pose[3],pose[4],pose[5])
        base_coords = np.dot(R,end_coords) + t 
        return base_coords