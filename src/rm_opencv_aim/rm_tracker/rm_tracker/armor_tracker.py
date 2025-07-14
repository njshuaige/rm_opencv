import math 
import numpy as np 

RAD2DEG = 180/math.pi#角度转化--弧度
DEG2RAD = math.pi/180

class Armor:
    def __init__(self, class_id, center, height, number=None, pose=None, armor_type=None):
        self.class_id = class_id#初始化定义
        self.center = center  
        self.height = height
        self.number = number  
        self.pose = pose     
        self.type = armor_type
class EKF:
    def __init__(self, dt=0.1):
        self.n = 9  # 状态维度
        self.m = 4  # 观测维度
        self.dt = dt

        self.x = np.zeros((self.n, 1))
        self.P = np.eye(self.n) * 1e3

        # 状态转移矩阵 (动态更新)
        self.F = np.eye(self.n)

        # 观测矩阵
        self.H = np.zeros((self.m, self.n))
        self.H[0, 0] = 1
        self.H[1, 2] = 1
        self.H[2, 4] = 1
        self.H[3, 6] = 1

        # 过程噪声和观测噪声
        self.Q = np.eye(self.n) * 1e-2
        self.R = np.eye(self.m) * 1e-1
        self.I = np.eye(self.n)

    def set_state(self, state):
        self.x = np.reshape(state, (self.n, 1))

    def build_F(self):
        F = np.eye(self.n)
        dt = self.dt
        F[0, 1] = dt  # x
        F[2, 3] = dt  # y
        F[4, 5] = dt  # z
        F[6, 7] = dt  # yaw
        self.F = F

    def predict(self):
        self.build_F()
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x.flatten()

    def update(self, z):
        z = np.reshape(z, (self.m, 1))
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)
        return self.x.flatten()

    def set_Q(self, q_scale):
        self.Q = np.eye(self.n) * q_scale

    def set_R(self, r_scale):
        self.R = np.eye(self.m) * r_scale


class Tracker:
    LOST = 0
    DETECTING = 1
    TRACKING = 2
    TEMP_LOST = 3
    CHANGE_TARGET = 4

    def __init__(self, max_match_distance, max_match_yaw_diff):
        self.tracker_state = self.LOST
        self.tracked_id = ""
        self.measurement = np.zeros(4)
        self.target_state = np.zeros(9)
        self.max_match_distance_ = max_match_distance
        self.max_match_yaw_diff_ = max_match_yaw_diff

        self.ekf = EKF(dt=0.1)

        self.last_yaw_ = 0.0
        self.change_count_ = 0
        self.change_thres = 20
        self.detect_count_ = 0
        self.tracking_thres = 5
        self.lost_count_ = 0
        self.lost_thres = 5

        self.tracked_armor = None
        self.another_r = 0
        self.dz = 0

    def init_ekf(self, init_state):
        self.ekf.set_state(init_state)
        self.target_state = init_state.copy()

    def predict(self):
        self.target_state = self.ekf.predict()

    def update(self, measurement):
        self.target_state = self.ekf.update(measurement)

    def set_noise(self, q_scale, r_scale):
        self.ekf.set_Q(q_scale)
        self.ekf.set_R(r_scale)



def select_tracking_armor(msg, color, track_height_tol):
    """
    从 ArmorsCppMsg 中选择目标装甲板。
    
    Args:
        msg: ArmorsCppMsg 类型的消息
        color: 目标颜色 (1: 蓝色, 0: 红色)
        track_height_tol: 高度差阈值
        cx_tol: 中心 x 坐标差阈值
    
    Returns:
        list: 包含单个目标 Armor 对象的列表，若无目标则为空列表
    """
    # 解包装甲板信息
    armor_info = [Armor(info.class_id, (info.cx, info.cy), info.height) for info in msg.armors]
    
    # 如果没有装甲板，直接返回空列表
    if not armor_info:
        return []

    # 根据颜色筛选：1 表示蓝色 (class_id < 6)，0 表示红色 (class_id > 5)
    if color == 1:
        filtered_color_data = [armor for armor in armor_info if armor.class_id < 6]
    elif color == 0:
        filtered_color_data = [armor for armor in armor_info if armor.class_id > 5]
    else:
        return []  # 非法颜色值

    # 如果筛选后没有装甲板，返回空列表
    if not filtered_color_data:
        return []

    # 如果只有一个装甲板，直接返回
    if len(filtered_color_data) == 1:
        return filtered_color_data

    # 按高度排序，取前两个
    top_two = sorted(filtered_color_data, key=lambda armor: armor.height, reverse=True)[:2]
    height_diff = top_two[0].height - top_two[1].height

    # 高度差不多，就返回 x 坐标最小的装甲板
    if height_diff <= track_height_tol:
        return [top_two[0]] if abs(top_two[0].center[0]) < abs(top_two[1].center[0]) else [top_two[1]]

    # 否则，返回 y 坐标最高的装甲板
    return [top_two[0]]


                    