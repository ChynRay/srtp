import numpy as np
from piper_sdk import kinematics
class PiperAnalyticalIK:
    def __init__(self,dh_type):
        # DH参数, 单位: mm 和 rad
        self.a1 = 0.0
        self.a2 = 285.03
        self.a3 = -21.984
        self.a4 = 0.0
        self.a5 = 0.0
        self.a6 = 0.0

        self.alpha1 = -np.pi / 2.0
        self.alpha2 = 0.0
        self.alpha3 = np.pi / 2.0
        self.alpha4 = -np.pi / 2.0
        self.alpha5 = np.pi / 2.0
        self.alpha6 = 0.0

        self.d1 = 123
        self.d2 = 0.0
        self.d3 = 0.0
        self.d4 = 250.75
        self.d5 = 0.0
        self.d6 = 91 + 0
        self.dhtype = dh_type

        # 关节角度偏移补偿
        self.theta_offset = [0.0, -172.22, -102.78, 0.0, 0.0, 0.0]

        # 关节角度限制, 单位: rad
        # self.joint_limits = [
        #     [-154, 154],  # joint1: ±154°
        #     [0, 195],                # joint2: 0~195°
        #     [-175, 1],               # joint3: -175~0°
        #     [-102, 102],   # joint4: ±102°
        #     [-75, 75],     # joint5: ±75°
        #     [-120, 120]    # joint6: ±120°
        # ]
        self.joint_limits = [
            [-154, 154],  # joint1: ±154°
            [0, 195],                # joint2: 0~195°
            [-175, 1],               # joint3: -175~0°
            [-110, 110],   # joint4: ±102°
            [-80, 80],     # joint5: ±75°
            [-120, 120]    # joint6: ±120°
        ]

    def cal_wristcenter(self,p6,z6):
        """
        *计算机械臂腕轴位置*  

        :param p6: 末端执行器位置  
        :param z6: 末端执行器z轴方向  
        :return: 腕轴位置
        """
        P_wrist = p6 - self.d6 * z6
        return P_wrist

    def cal_j1j2j3(self,P_wrist):
        """
        *计算机械臂前三个关节角度*  

        :param P_wrist: 腕轴位置    
        :return: 关节1,2,3的角度
        """
        solutions = [] #储存角度的解
        x,y,z = P_wrist
        #计算j1
        j1 = np.atan2(y,x)
        print(j1)
        #调整基准高度
        z = z - self.d1
        # x1 = x * np.cos(j1) + y * np.sin(j1)
        # r1 = np.sqrt(x1 ** 2 + z ** 2)
        #计算关节1，2，3角度
        r = np.sqrt(x**2+y**2)
        L2 = self.a2
        L3 = np.sqrt(self.a3**2+self.d4**2)
        D = (r**2+z**2-L2**2-L3**2)/(2*L2*L3)
        if abs(D) > 1:
            print("无解:D={}".format(D)) #去除不符合条件的值
            return solutions
        phi = np.atan2(self.d4,abs(self.a3))
        beta = np.acos(D)
        beta_d = np.pi - beta

        #两种可能的j3值
        for i in [-1,1]:
            for m in [-1,1]:
                j3 = i * beta_d - m * phi
            for j in [-1,1]:
                #计算j2
                K1 = L2 + L3 *abs(np.cos(beta))*j            
                K2 = abs(L3 * np.sin(beta))                 
                #geo            
                gamma = np.atan2(z,r)            
                delta = np.atan2(K2,K1)
                j2 = gamma - delta
                for x in [-1,0,1]:
                    for y in [-2,-1,0,1,2]:
                        j1_c = j1 + x * np.pi
                        j3_c = j3 + y * np.pi
                        solutions.append([j1_c,j2,j3_c])
                j2 = np.pi - gamma + delta
                for x in [-1,0,1]:
                    for y in [-2,-1,0,1,2]:
                        j1_c = j1 + x * np.pi
                        j3_c = j3 + y * np.pi
                        solutions.append([j1_c,j2,j3_c])
                j2 = np.pi - gamma - delta
                for x in [-1,0,1]:
                    for y in [-2,-1,0,1,2]:
                        j1_c = j1 + x * np.pi
                        j3_c = j3 + y * np.pi
                        solutions.append([j1_c,j2,j3_c])
                j2 = (gamma + delta)
                # j2 = np.pi  - j2
                #ansys
                # K = np.sqrt(K1**2 + K2**2)
                # cos_j2 = (K1*r + K2*z)/(K**2)
                # sin_j2 = (-K2*r + K1*z)/(K**2)
                # j2 = np.atan2(sin_j2,cos_j2)
                # j2 = np.acos(cos_j2)
                # j2 = np.asin(sin_j2)
                for x in [-1,0,1]:
                    for y in [-2,-1,0,1,2]:
                        j1_c = j1 + x * np.pi
                        j3_c = j3 + y * np.pi
                        solutions.append([j1_c,j2,j3_c])
        return solutions

    def cal_j4j5j6(self, arm_joints, R_target):
        """
        *计算j4,j5,j6*  

        :param arm_joints: 机械臂关节角度
        :param R_target: 末端旋转矩阵
        :return: 关节4,5,6的角度
        """

        solutions = []
        for i in range(len(arm_joints)):
            q1, q2, q3 = arm_joints[i][0], arm_joints[i][1], arm_joints[i][2]

            # 计算基座到关节3的旋转矩阵
            R03 = self.compute_R03(q1, q2, q3)
            # 计算关节3到末端的期望旋转矩阵
            R36 = R03.T @ R_target
            # 使用ZYZ欧拉角进行解算，多解性小，适合机械臂
            r11, r12, r13 = R36[0,0], R36[0,1], R36[0,2]
            r21, r22, r23 = R36[1,0], R36[1,1], R36[1,2]
            r31, r32, r33 = R36[2,0], R36[2,1], R36[2,2]

            # ZYZ欧拉角解算q4/q5/q6，双解分支
            for sign in [1.0, -1.0]:
                # 奇异位姿处理：q5=0 或 π
                if abs(r33) > 0.9999:
                    q5 = 0.0 if r33 > 0 else np.pi
                    q4 = 0.0  # 奇异位姿q4任意，默认0
                    q6 = np.atan2(r21, r11)
                    solutions.append([q4, q5, q6])
                else:
                    # 正常解算逻辑
                    q5 = sign * np.acos(r33)
                    q4 = np.atan2(r23, r13)
                    q6 = np.atan2(r32, -r31)

                    # 角度一致性修正
                    if sign < 0:
                        q4 += np.pi
                        q6 -= np.pi

                    # 角度归一化：atan2(sinθ,cosθ) → [-π, π]
                    q4 = np.atan2(np.sin(q4), np.cos(q4))
                    q6 = np.atan2(np.sin(q6), np.cos(q6))
                    # solutions.append([q4,q5,q6])
                for i in [1,-1]:
                    solutions.append([q4,q5,i*q6])

        return solutions

    def compute_R03(self, theta1, theta2, theta3): 
        """
        *计算基座到关节3的旋转矩阵*
        
        :param theta1: 机械臂关节1角度
        :param theta2: 机械臂关节2角度
        :param theta3: 机械臂关节3角度
        :return: 基座到关节3的旋转矩阵
        """
        # 叠加三个关节的DH变换矩阵
        if self.dhtype == 'modified': #判断变换矩阵类型
            #改进DH矩阵变换
            T01 = self.modified_dh_transform(self.alpha1, self.a1, self.d1, theta1 + self.theta_offset[0])
            T12 = self.modified_dh_transform(self.alpha2, self.a2, self.d2, theta2 + self.theta_offset[1])
            T23 = self.modified_dh_transform(self.alpha3, self.a3, self.d3, theta3 + self.theta_offset[2])
        
        else:
            #标准DH矩阵变换
            T01 = self.standard_dh_transform(self.alpha1, self.a1, self.d1, theta1 + self.theta_offset[0])
            T12 = self.standard_dh_transform(self.alpha2, self.a2, self.d2, theta2 + self.theta_offset[1])
            T23 = self.standard_dh_transform(self.alpha3, self.a3, self.d3, theta3 + self.theta_offset[2])

        # 矩阵乘法得到T03，取前3x3为旋转矩阵
        T03 = T01 @ T12 @ T23
        return T03[:3, :3]

    # 改进型DH变换矩阵
    def modified_dh_transform(self, alpha, a, d, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        T = np.array([
            [cos_theta,        -sin_theta,         0,          a],
            [sin_theta*cos_alpha, cos_theta*cos_alpha, -sin_alpha, -sin_alpha*d],
            [sin_theta*sin_alpha, cos_theta*sin_alpha,  cos_alpha,  cos_alpha*d],
            [0,                0,                  0,          1]
        ])
        return T

    # 标准DH变换矩阵
    def standard_dh_transform(self, alpha: float, a: float, d: float, theta: float) -> np.ndarray:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        T = np.array([
            [cos_theta, -sin_theta*cos_alpha, sin_theta*sin_alpha, a*cos_theta],
            [sin_theta,  cos_theta*cos_alpha, -cos_theta*sin_alpha, a*sin_theta],
            [0,          sin_alpha,           cos_alpha,           d],
            [0,          0,                   0,                   1]
        ], dtype=np.float64)
        return T
    
    def apply_offset(self,solution):
        """
        *应用补偿*
        
        :param solution: 解算的角度
        :return: 补偿后的解
        """
        solution_c = solution[:]
        for i in range(len(solution_c)):
            solution_c[i] -= self.theta_offset[i]
        return solution_c
    
    def boolwithinlimits(self,solutions):
        """
        *判断角度是否超限*
        
        :param solutions: 所有解集合
        :return: 满足角度限制的解
        """
        solutions_ex = []
        for  i in range(len(solutions)):
            flags =[]
            solution = solutions[i]
            # if len(solution) != len(self.joint_limits):
            #     print('error:解得角度个数不符')
            for j in range(len(solution)):
                if solution[j] >= self.joint_limits[j][0] and solution[j] <= self.joint_limits[j][1]:
                    flags.append(1)
                else:
                    flags.append(0)
            # if flags == [1,1,1,1,1,1]:
            if flags == [1 for i in range(len(solution))]:
                solutions_ex.append(solution)
            # for n in range(len(flags)):
            #     if flags[n] == 0:
            #         print("第{0}组解中,第{1}个关节超限".format(i+1,n+1))
        return solutions_ex

def combine(j1j2j3,j4j5j6):
    #组合解
    solutions = []
    n = len(j1j2j3)
    m = len(j4j5j6)
    if n > 0:
        h = m // n
    for i in range(n):
        for j in range(h):
            solutions.append(j1j2j3[i]+j4j5j6[h*i+j])
    return solutions

def rad2deg(solutions_ex):
    solutions_exc = solutions_ex[:]
    for i in range(len(solutions_ex)):
        solutions_exc[i] = [solutions_ex[i][j]*180 / np.pi for j in range(len(solutions_ex[i]))]
    return solutions_exc
    

def deg2rad(solutions_ex):
    solutions_exc = solutions_ex[:]
    for i in range(len(solutions_ex)):
        solutions_exc[i] = [solutions_exc[i][j]/180 * np.pi for j in range(len(solutions_exc[i]))]
    return solutions_exc

def normalize_angle(angle):
        """角度归一化到 [-π, π]"""
        while angle >= 180:
            angle -= 180
            break
        while angle <= -180:
            angle += 180
            break
        return angle

def show_args(solutions_ex):
    n = len(solutions_ex)
    print('一共符合条件的解有{}组'.format(n))
    # print('解如下所示')
    # for i in range(n):
    #     print("第{0}组解为{1}".format(i+1,solutions_ex[i]))

def get_eulertf(rx,ry,rz):
    '''
    根据欧拉角获取末端坐标系变换后的三个轴方向向量
    输入:末端坐标系下欧拉角
    '''
    rx = np.radians(rx)
    ry = np.radians(ry)
    rz = np.radians(rz)
    cr, sr = np.cos(rx), np.sin(rx)
    cp, sp = np.cos(ry), np.sin(ry)
    cy, sy = np.cos(rz), np.sin(rz)

    # 构建Z-Y-X内旋的旋转矩阵
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr]
    ])
    
    # 第二步：提取旋转矩阵的列向量作为三轴方向（核心！）
    x_axis = R[:, 0]  # 第一列：X轴方向向量
    y_axis = R[:, 1]  # 第二列：Y轴方向向量
    z_axis = R[:, 2]  # 第三列：Z轴方向向量
    
    # 浮点精度修正（确保单位向量，避免微小误差）
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    return R

# cal_ik = PiperAnalyticalIK(dh_type="st")
# # p_total = [56.128, 0.0, 213.266, 0.0, 85.0,0.0]
# # p_total = [391.736,12.811,192.621,-178.999,-0.919,-100.638] #单位mm和°
# p_total = [348.375,-29.391,412.241,129.653,85.9,124.777]
# # p_total = [387.131,6.649,185.211,177.549,-1.36,171.520]
# p = p_total[0:3]
# R = get_eulertf(p_total[3],p_total[4],p_total[5])
# z = R[:,2]
# p_wrist = cal_ik.cal_wristcenter(p,z)
# j1j2j3s = cal_ik.cal_j1j2j3(p_wrist)
# j1j2j3s_deg = rad2deg(j1j2j3s)
# j1j2j3s_off = [cal_ik.apply_offset(j1j2j3s_deg[j]) for j in range(len(j1j2j3s_deg))]
# j1j2j3s_n = []
# for i in range(len(j1j2j3s_off)):
#     j1j2j3_n = [normalize_angle(j1j2j3s_off[i][j]) for j in range(len(j1j2j3s_off[i]))]
#     j1j2j3s_n.append(j1j2j3_n)
# j1j2j3s_ex = cal_ik.boolwithinlimits(j1j2j3s_n)
# j1j2j3s_exrad = deg2rad(j1j2j3s_ex)
# j4j5j6s = cal_ik.cal_j4j5j6(j1j2j3s_exrad,R)
# j4j5j6s_deg = rad2deg(j4j5j6s)
# ss = combine(j1j2j3s_ex,j4j5j6s_deg)
# ss_ex = cal_ik.boolwithinlimits(ss)
# # show_args(ss_ex)
# #FK正向运动验证,剔除不合理的解
# cal_fk = kinematics.C_PiperForwardKinematics()
# ss_final = []
# for i in range(len(ss_ex)):
#     ss1 = [j/180 * np.pi for j in ss_ex[i]]
#     ps = cal_fk.CalFK(ss1)[-1]
#     print(ps)
#     flags = []
#     for j in range(len(ps)):
#         if abs(abs(ps[j])-abs(p_total[j])) < 10:
#             flags.append(1)
#     if flags == [1,1,1,1,1,1]:
#         ss_final.append(ss_ex[i])
#         print('预期:{}'.format(p_total))
#         print('计算:{}'.format(ps))
# show_args(ss_final)

# #[2.132,114.492,-85.854,-1.227,65.595,76.736]
# #[-4.829,83.665,-75.570,-3.17,-0.470,6.327]

def posetoangle(p_total):
    cal_ik = PiperAnalyticalIK(dh_type="st")
    p = p_total[0:3]
    R = get_eulertf(p_total[3],p_total[4],p_total[5])
    z = R[:,2]
    p_wrist = cal_ik.cal_wristcenter(p,z)
    j1j2j3s = cal_ik.cal_j1j2j3(p_wrist)
    j1j2j3s_deg = rad2deg(j1j2j3s)
    j1j2j3s_off = [cal_ik.apply_offset(j1j2j3s_deg[j]) for j in range(len(j1j2j3s_deg))]
    j1j2j3s_n = []
    for i in range(len(j1j2j3s_off)):
        j1j2j3_n = [normalize_angle(j1j2j3s_off[i][j]) for j in range(len(j1j2j3s_off[i]))]
        j1j2j3s_n.append(j1j2j3_n)
    j1j2j3s_ex = cal_ik.boolwithinlimits(j1j2j3s_n)
    j1j2j3s_exrad = deg2rad(j1j2j3s_ex)
    j4j5j6s = cal_ik.cal_j4j5j6(j1j2j3s_exrad,R)
    j4j5j6s_deg = rad2deg(j4j5j6s)
    ss = combine(j1j2j3s_ex,j4j5j6s_deg)
    ss_ex = cal_ik.boolwithinlimits(ss)
    # show_args(ss_ex)
    #FK正向运动验证,剔除不合理的解
    cal_fk = kinematics.C_PiperForwardKinematics()
    ss_final = []
    for i in range(len(ss_ex)):
        ss1 = [j/180 * np.pi for j in ss_ex[i]]
        ps = cal_fk.CalFK(ss1)[-1]
        # ps[0:3] = ps[0:3] + z*0
        # print(ps)
        flags = []
        ps1 = ps[0:3]
        ps2 = ps[3:]
        for j in range(len(ps1)):
            if abs(ps1[j]-p_total[j]) < 5 and abs(abs(ps2[j])-abs(p_total[j+3])) < 5:
                flags.append(1)
        if flags == [1,1,1]:
            ss_final.append(ss_ex[i])
            print('预期:{}'.format(p_total))
            print('计算:{}'.format(ps))
    show_args(ss_final)
    if len(ss_final) > 0:
        return ss_final[0]
    else:
        return None
    # return ss_final
# p = [-26.63651174124989, -56.53039858760961, 434.67724577172044, 100.10193202455119, -8.95311391875667, 111.11673808853797]
# posetoangle(p)

