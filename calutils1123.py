import numpy as np
import cv2
import open3d as o3d
import math

def get_normal(x,y,z):
    '''
    从空间不共线三点组成的平面求其法向量
    输入：三点坐标 (向量形式)
    '''
    p1 = np.array(x)
    p2 = np.array(y)
    p3 = np.array(z)
    # 计算法向量
    normal = np.cross(p2 - p1, p3 - p1)
    normal = normal / np.linalg.norm(normal)  # 归一化
    if normal[2] >= 0:
        normal = -normal
    return normal

def get_xaxil(y,z):
    '''
    获取相机坐标系下x轴方向
    输入y轴与z轴方向向量
    '''
    p1 = np.array(y)
    p2 = np.array(z)
    xaxil = np.cross(p1, p2)
    xaxil = xaxil / np.linalg.norm(xaxil)  # 归一化
    return xaxil

def get_tfH(x,y,z):
    '''
    获取末端坐标系变换后的旋转矩阵
    输入:末端坐标系下x,y,z轴方向向量
    '''
    H = np.zeros((3,3))
    H[0,0]=x[0]
    H[1,0]=x[1]
    H[2,0]=x[2]
    H[0,1]=y[0]
    H[1,1]=y[1]
    H[2,1]=y[2]
    H[0,2]=z[0]
    H[1,2]=z[1]
    H[2,2]=z[2]
    return H

def get_tfeuler(x,y,z):
    '''
    获取末端坐标系变换后的欧拉角
    输入:末端坐标系下x,y,z轴方向向量
    '''
    H = get_tfH(x,y,z)
    yaw = math.atan2(H[1,0], H[0,0])
    pitch = math.atan2(-H[2,0],math.hypot(H[0,0],H[1,0]))
    roll = math.atan2(H[2,1], H[2,2])
    [rx,ry,rz] =np.degrees([roll, pitch,yaw])
    return [rx,ry,rz]

def get_eulertf(rx,ry,rz):
    '''
    根据欧拉角获取末端坐标系变换后的旋转矩阵
    输入:末端坐标系下欧拉角
    '''
    rx = np.radians(rx)
    ry = np.radians(ry)
    rz = np.radians(rz)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

if __name__ == '__main__':
    meuler = get_tfeuler([1,1,0],[-1,1,0],[0,0,1])
    normal = get_normal([1,1,0],[-1,1,0],[0,0,1])
    print(normal)