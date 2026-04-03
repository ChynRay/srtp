from piper_sdk import * # type: ignore
import time
import numpy as np # type: ignore

class robot_arm:
    def __init__(self,can_port_name='can0'):
        """控制piper机械臂

        Parameters
        ----------
        can_port_name : str, optional
            _description_, by default 'can0'
        """
        self.piper = C_PiperInterface_V2(can_port_name) # type: ignore
        self.piper.ConnectPort()
    def enable_arm(self,enable):         #使能或禁用机械臂
        """使能机械臂
        :param enable: bool True 为使能，False为禁用
        """

        enable_flag = False
        loop_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        elapsed_time_flag = False
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            print(f"--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if(enable):
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0,1000,0x01, 0)
            else:
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0,1000,0x02, 0)
            print(f"使能状态: {enable_flag}")
            print(f"--------------------")
            if(enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else: 
                loop_flag = False
                enable_flag = False
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print(f"超时....")
                elapsed_time_flag = True
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        print(f"Returning response: {resp}")
        return resp
    def move_arm_points(self,position,factor=1000):
        """
        通过末端位置控制机械臂

        :param position: list 末端位置 [x,y,z,rx,ry,rz]
        :param factor: int 放大倍数
        """

        # self.piper.EnableArm(7)
        # self.piper.GripperCtrl(0,1000,0x01, 0)
        X = round(position[0]*factor)
        Y = round(position[1]*factor)
        Z = round(position[2]*factor)
        RX = round(position[3]*factor)
        RY = round(position[4]*factor)
        RZ = round(position[5]*factor)
        # joint_6 = round(position[6]*factor)
        print('set point position')
        print(X,Y,Z,RX,RY,RZ)
        # piper.MotionCtrl_1()
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)   #设置运动模式为末端位置控制
        self.piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)            #设置末端位置
        time.sleep(5) 
        
        x,y,z,rx,ry,rz= self.read_point_position()

    

        print('end point position')
        print(x,y,z,rx,ry,rz)
        self.check_error()         #检查机械臂状态
        
            
        # self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
    def move_arm_joints(self,position,factor=1000):
        """
        通过关节角度控制机械臂

        :param position: list 关节角度 [j1,j2,j3,j4,j5,j6]
        :param factor: int 放大倍数
        """

        # self.piper.EnableArm(7)
        # self.piper.GripperCtrl(0,1000,0x01, 0)
        joint_0 = round(position[0]*factor)
        joint_1 = round(position[1]*factor)
        joint_2 = round(position[2]*factor)
        joint_3 = round(position[3]*factor)
        joint_4 = round(position[4]*factor)
        joint_5 = round(position[5]*factor)

        # piper.MotionCtrl_1()
        print('set joint position')
        print(joint_0,joint_1,joint_2,joint_3,joint_4,joint_5)
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)  #设置运动模式为关节角度控制
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5) #    设置关节角度
        time.sleep(5)
        
        j1,j2,j3,j4,j5,j6= self.read_joint_position()

        print('end joint position')
        print(j1,j2,j3,j4,j5,j6)
        self.check_error()  # 检查机械臂状态
    def gripper_ctrl(self,range,factor=1000):
        """
        控制机械臂夹爪

        :param position: int 夹爪位置
        :param factor: int 放大倍数
        """

        self.piper.GripperCtrl(0,1000,0x01, 0)
        self.piper.GripperCtrl(abs(range)*factor, 1000, 0x01, 0)
    def return_to_base_position(self): #将机械臂回到基座位置
        base_position=[54.952,0,203.386,0,85,0]
        base_joints=[0.0,0.0,0.0,0.0,0.0,0.0]
        self.move_arm_joints(base_joints)

    def read_point_position(self):  #读取末端位置
        endpose=self.piper.GetArmEndPoseMsgs().end_pose
        x,y,z,rx,ry,rz= endpose.X_axis,endpose.Y_axis,endpose.Z_axis,endpose.RX_axis,endpose.RY_axis,endpose.RZ_axis
        return x,y,z,rx,ry,rz
    
    def read_joint_position(self): #读取关节角度
        end_joint=self.piper.GetArmJointMsgs().joint_state
        j1,j2,j3,j4,j5,j6=end_joint.joint_1,end_joint.joint_2,end_joint.joint_3,end_joint.joint_4,end_joint.joint_5,end_joint.joint_6
        return j1,j2,j3,j4,j5,j6
   

    def read_arm_status(self):  #读取机械臂状态
        all_status={}
        all_arm_status=self.piper.GetArmStatus().arm_status
        print(all_arm_status)
        arm_status=all_arm_status.arm_status
        motion_status=all_arm_status.motion_status
        joint_limit_status={}
        
        joint_limit_status[f'joint1']=all_arm_status.err_status.joint_1_angle_limit
        joint_limit_status[f'joint2']=all_arm_status.err_status.joint_2_angle_limit
        joint_limit_status[f'joint3']=all_arm_status.err_status.joint_3_angle_limit
        joint_limit_status[f'joint4']=all_arm_status.err_status.joint_4_angle_limit
        joint_limit_status[f'joint5']=all_arm_status.err_status.joint_5_angle_limit
        joint_limit_status[f'joint6']=all_arm_status.err_status.joint_6_angle_limit
        
        all_status['arm_status']=arm_status
        all_status['motion_status']=motion_status
        all_status['joint_limit_status']=joint_limit_status
        self.all_arm_status=all_status
    def read_motor_status(self): #读取电机状态
        def read_single_motor_status(motor_status):
            sigle_motor_status={}
            sigle_motor_status['voltage_too_low']=motor_status.voltage_too_low
            sigle_motor_status['motor_overheating']=motor_status.motor_overheating
            sigle_motor_status['driver_overcurrent']=motor_status.driver_overcurrent
            sigle_motor_status['driver_overheating']=motor_status.driver_overheating
            sigle_motor_status['collision_status']=motor_status.collision_status
            sigle_motor_status['stall_status']=motor_status.stall_status
            return sigle_motor_status
        all_motor_status={}
        motor_status=self.piper.GetArmLowSpdInfoMsgs()
        # print(motor_status)
        motor1_status=motor_status.motor_1.foc_status
        motor2_status=motor_status.motor_2.foc_status
        motor3_status=motor_status.motor_3.foc_status
        motor4_status=motor_status.motor_4.foc_status
        motor5_status=motor_status.motor_5.foc_status
        motor6_status=motor_status.motor_6.foc_status
        all_motor_status['motor1']=read_single_motor_status(motor1_status)
        all_motor_status['motor2']=read_single_motor_status(motor2_status)
        all_motor_status['motor3']=read_single_motor_status(motor3_status)
        all_motor_status['motor4']=read_single_motor_status(motor4_status)
        all_motor_status['motor5']=read_single_motor_status(motor5_status)
        all_motor_status['motor6']=read_single_motor_status(motor6_status)
        self.all_motor_status=all_motor_status

    def remove_error(self,joint_num): #清除电机错误
        '''
        [0x00, 0xAE] 第一个不处理，第二个清除错误
        '''
        self.piper.JointConfig(joint_num=joint_num, clear_err=0xAE)
    def set_zero_position(self,joint_num): #设置电机零位

        self.piper.JointConfig(joint_num=joint_num, set_zero=0xAE)

    def check_error(self):  #检查机械臂状态
        def check_motor_error(motor_status,motor_id):
            motor_infor=motor_status['motor'+str(motor_id)]
            for value in motor_infor.values():
                if value==1:
                    return True
            return False
        self.read_arm_status()
        motor_list=[1,2,3,4,5,6]
        error_motor_list=[]
        self.read_arm_status()
        if self.all_arm_status['arm_status']!=0:
            print(self.all_arm_status['arm_status'])
            print(self.all_arm_status['motion_status'])
            print(self.all_arm_status['joint_limit_status'])
            self.read_motor_status()
            for motor_id in motor_list:
                if check_motor_error(self.all_motor_status,motor_id):
                    print(self.all_motor_status['motor'+str(motor_id)])
                    error_motor_list.append(motor_id)
            if len(error_motor_list)>0:
                self.error_process(error_motor_list)
            else:
                print('未知错误')
        else:
            print('无错误')
    def error_process(self,error_motor_list):  #处理机械臂错误
        flag=input('是否准备好失能？(y/n)')
        if flag=='y':
            self.piper.EnableArm(False)
            for id in error_motor_list:
                self.remove_error(id)
        else:
            pass


 