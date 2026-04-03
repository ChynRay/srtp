# 纯机械臂控制，不需要标定文件！
import time
from utils.piper_arm import robot_arm

class RobotArm:
    def __init__(self):
        self.arm = robot_arm()

    def enable(self):
        self.arm.enable_arm(True)
        time.sleep(0.5)

    def disable(self):
        self.arm.enable_arm(False)

    def move_joints(self, angles):
        self.enable()
        self.arm.move_arm_joints(angles)

    def move_home(self):
        self.move_joints([0,0,0,0,0,0])

    # def gripper_open(self):
    #     self.enable()
    #     # 你的夹爪代码

    # def gripper_close(self):
    #     self.enable()
    #     # 你的夹爪代码