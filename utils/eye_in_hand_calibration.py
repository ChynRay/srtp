import cv2
import numpy as np
import transforms3d as tfs
import math
import csv
def EulerAngle_to_R(rxryrz):
    '''
    角度变换转化为旋转矩阵变换
    rxryrz为array类型
    '''
    rx,ry,rz = rxryrz
    R = tfs.euler.euler2mat(math.radians(rx), math.radians(ry), math.radians(rz), axes='sxyz')  # 注意欧拉角顺序
    return R 

def Rvec_to_R(rvec):
    '''
    旋转向量变换转旋转矩阵变换
    rvec为array类型
    '''
    # 使用Rodrigues公式将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    # 返回旋转矩阵
    return R

def xyz_to_t(xyz):
    '''xyz转平移向量t'''
    t = xyz.reshape(3, 1)
    return t

def Rt_to_H(R,t):
    '''旋转矩阵R和平移向量t结合成齐次矩阵H'''
    H = tfs.affines.compose(np.squeeze(t), R, [1, 1, 1])
    return H

def handineye_calibration(end2base_xyzrxryrz,board2cam_xyzrxryrz):
    """
    说明：
        利用机械臂末端位姿arrry：end2base_xyzrxryrz和
        标定板位姿arrry：board2cam_xyzrxryrz，
        使用cv2.calibrateHandEye()函数（默认为Tsai方法计算AX=XB）
        来获得相机到机械臂基底的变换矩阵cam2end_H

    Args:
        end2base_xyzrxryrz: 机械臂末端位姿arrry
        board2cam_xyzrxryrz: 标定板位姿arrry

    Returns:
        cam2end_H: 相机到机械臂末端的齐次变换矩阵
    """
    end2base_R , end2base_t =  [], []
    board2cam_R , board2cam_t =  [], []

    # end2base_xyzrxryrz转换为end2base_R和end2base_t
    for xyzrxryrz in end2base_xyzrxryrz: # xyzrxryrz为array类型
        R = EulerAngle_to_R(xyzrxryrz[3:6])
        t = xyz_to_t(xyzrxryrz[0:3])
        end2base_R.append(R)
        end2base_t.append(t)

    # board2cam_xyzrxryrz转换为board2cam_R和board2cam_t
    for xyzrxryrz in board2cam_xyzrxryrz:# xyzrxryrz为array类型
        R = Rvec_to_R(xyzrxryrz[3: 6])
        t = xyz_to_t(xyzrxryrz[0:3])
        board2cam_R.append(R)
        board2cam_t.append(t)

    # end2base转换为base2end
    base2end_R, base2end_t = [], []
    for R, t in zip(end2base_R, end2base_t):
        R_b2e = R.T
        t_b2e = -R_b2e @ t
        base2end_R.append(R_b2e)
        base2end_t.append(t_b2e)

    # 使用cv2.calibrateHandEye()函数计算cam2end_R和cam2end_t
    cam2end_R, cam2end_t = cv2.calibrateHandEye(  end2base_R,
                                                  end2base_t,
                                                  board2cam_R,
                                                  board2cam_t,
                                                  method=cv2.CALIB_HAND_EYE_HORAUD)  # TSAI PARK DANIILIDIS HORAUD ANDREFF
    cam2end_H = Rt_to_H(cam2end_R,cam2end_t) #合成齐次矩阵H

    return cam2end_H #返回齐次矩阵H


def main(mode='R'):
    print("========读取end2base_xyzrxryrz和board2cam_xyzrxryrz测试手眼标定获得cam2end_H=========")
    with open('end2base_xyzrxryrz.csv', newline='') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',')
        end2base_xyzrxryrz = [list(map(float, row)) for row in data_reader]
    end2base_xyzrxryrz = np.array(end2base_xyzrxryrz)

    with open(f'board2cam_xyzrxryrz_{mode}.csv', newline='') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',')
        board2cam_xyzrxryrz = [list(map(float, row)) for row in data_reader]
    board2cam_xyzrxryrz = np.array(board2cam_xyzrxryrz)

    cam2end_H = handineye_calibration(end2base_xyzrxryrz,board2cam_xyzrxryrz)

    # with open('../src/output/cam2end_H.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(cam2end_H)
    print("cam2base_H:\n{}".format(cam2end_H)) #打印齐次矩阵H


if __name__ == '__main__':
    main(mode="R")
    main(mode="L")
    #mode="R"表示右手标定，mode="L"表示左手标定
    # 右手标定和左手标定的区别在于末端位姿的旋转角度不同，
    # 右手标定的末端位姿旋转角度为[0, 0, 0, 0, 0, 0]，
    # 左手标定的末端位姿旋转角度为[0, 0, 0, 180, 0, 0]，
    # 其他参数相同。