# -*- coding: utf-8 -*-
"""
A题 问题1：FY1投放1枚烟幕干扰弹对M1的干扰
给定参数：FY1朝假目标方向120m/s飞行，1.5s后投放，3.6s后起爆
计算有效遮蔽时长
"""
import math
import numpy as np

G = 9.8  # 重力加速度 m/s^2

# ===================== 坐标与参数 =====================
# 假目标（原点）
DECOY = np.array([0.0, 0.0, 0.0])
# 真目标：圆柱体，半径7m，高10m，下底面圆心(0,200,0)
TARGET_CENTER = np.array([0.0, 200.0, 5.0])  # 圆柱中心
TARGET_R = 7.0
TARGET_H = 10.0
TARGET_BOTTOM = np.array([0.0, 200.0, 0.0])

# 导弹M1
M1_START = np.array([20000.0, 0.0, 2000.0])
M1_SPEED = 300.0
M1_DIST = np.linalg.norm(M1_START)

# FY1
FY1_START = np.array([17800.0, 0.0, 1800.0])
FY1_SPEED = 120.0
# 朝假目标方向：水平投影方向 = (-1, 0)
FY1_DIR = np.array([-1.0, 0.0, 0.0])

# 投放参数
T_RELEASE = 1.5   # 投放时刻
DT_FUSE = 3.6     # 引信延时（投放到起爆）
T_DETON = T_RELEASE + DT_FUSE  # 起爆时刻 = 5.1s

# 云团参数
CLOUD_R = 10.0    # 有效半径
CLOUD_SINK = 3.0  # 下沉速度
CLOUD_DURATION = 20.0  # 有效时长

# ===================== 运动学模型 =====================
def missile_pos(t):
    """导弹M1在时刻t的位置（匀速直线飞向原点）"""
    ratio = 1.0 - M1_SPEED * t / M1_DIST
    return M1_START * ratio

def uav_pos(t):
    """FY1在时刻t的位置（等高度匀速直线）"""
    return FY1_START + FY1_DIR * FY1_SPEED * t

def release_point():
    """投放点"""
    return uav_pos(T_RELEASE)

def detonation_point():
    """起爆点（平抛运动）"""
    r = release_point()
    dt = DT_FUSE
    d = np.array([
        r[0] + FY1_DIR[0] * FY1_SPEED * dt,
        r[1] + FY1_DIR[1] * FY1_SPEED * dt,
        r[2] - 0.5 * G * dt**2
    ])
    return d

def cloud_center(t):
    """云团中心在时刻t的位置（起爆后匀速下沉）"""
    d = detonation_point()
    if t < T_DETON:
        return None
    return np.array([d[0], d[1], d[2] - CLOUD_SINK * (t - T_DETON)])

# ===================== 目标代表点 =====================
def target_representative_points():
    """圆柱体表面的代表点（顶面和底面圆周 + 中心）"""
    points = []
    n_angles = 8
    for i in range(n_angles):
        ang = 2 * math.pi * i / n_angles
        # 底面圆周
        points.append(np.array([TARGET_R * math.cos(ang),
                                200.0 + TARGET_R * math.sin(ang),
                                0.0]))
        # 顶面圆周
        points.append(np.array([TARGET_R * math.cos(ang),
                                200.0 + TARGET_R * math.sin(ang),
                                TARGET_H]))
    # 中心点
    points.append(np.array([0.0, 200.0, 0.0]))
    points.append(np.array([0.0, 200.0, TARGET_H]))
    points.append(np.array([0.0, 200.0, TARGET_H / 2]))
    return points

# ===================== 视线遮蔽判定 =====================
def point_to_segment_dist(p, a, b):
    """点p到线段ab的最短距离，返回(距离, 最近点参数)"""
    ab = b - a
    ap = p - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-12:
        return np.linalg.norm(ap), 0.0
    t = np.dot(ap, ab) / ab_sq
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest), t

def is_obscurated(m_pos, c_pos, target_pts, cloud_r=CLOUD_R):
    """
    判断云团是否完全遮挡导弹到真目标的视线
    对所有目标代表点，检查线段M->T是否经过云团球体
    """
    for tp in target_pts:
        dist, seg_param = point_to_segment_dist(c_pos, m_pos, tp)
        if dist > cloud_r:
            return False
        # 还需检查交点是否在线段上（seg_param已在[0,1]内）
    return True

def is_obscurated_angular(m_pos, c_pos, target_pts, cloud_r=CLOUD_R):
    """
    角覆盖法：从导弹看云团的半角 vs 目标对云心方向的最大张角
    """
    d_mc = np.linalg.norm(c_pos - m_pos)
    if d_mc <= cloud_r:
        return True  # 导弹在云团内
    alpha = math.asin(min(1.0, cloud_r / d_mc))
    mc_dir = (c_pos - m_pos) / d_mc
    for tp in target_pts:
        mt = tp - m_pos
        d_mt = np.linalg.norm(mt)
        if d_mt < 1e-9:
            continue
        mt_dir = mt / d_mt
        cos_angle = np.dot(mc_dir, mt_dir)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)
        if angle > alpha:
            return False
    return True

# ===================== 主计算 =====================
def compute_obscuration_time():
    release = release_point()
    deton = detonation_point()
    print(f"投放点: ({release[0]:.2f}, {release[1]:.2f}, {release[2]:.2f})")
    print(f"起爆点: ({deton[0]:.2f}, {deton[1]:.2f}, {deton[2]:.2f})")
    print(f"起爆时刻: {T_DETON}s")
    print(f"有效窗口: [{T_DETON}, {T_DETON + CLOUD_DURATION}]s")
    print()

    target_pts = target_representative_points()
    print(f"目标代表点数: {len(target_pts)}")

    # 时间步进扫描
    dt = 0.001  # 步长 0.001s
    t_start = T_DETON
    t_end = T_DETON + CLOUD_DURATION

    obscuration_intervals = []
    in_obscuration = False
    interval_start = None

    method = "angular"  # "angular" or "point"

    t = t_start
    while t <= t_end:
        m = missile_pos(t)
        c = cloud_center(t)
        if c is None:
            t += dt
            continue

        if method == "angular":
            obscured = is_obscurated_angular(m, c, target_pts)
        else:
            obscured = is_obscurated(m, c, target_pts)

        if obscured and not in_obscuration:
            in_obscuration = True
            interval_start = t
        elif not obscured and in_obscuration:
            in_obscuration = False
            obscuration_intervals.append((interval_start, t))

        t += dt

    if in_obscuration:
        obscuration_intervals.append((interval_start, t_end))

    total_time = sum(b - a for a, b in obscuration_intervals)

    print(f"\n遮蔽判定方法: {method}")
    print(f"有效遮蔽时间区间:")
    for i, (a, b) in enumerate(obscuration_intervals):
        print(f"  区间{i+1}: [{a:.3f}, {b:.3f}]s, 时长 {b-a:.3f}s")
    print(f"\n总有效遮蔽时长: {total_time:.4f}s")

    # 用另一种方法验证
    method2 = "point"
    obscuration_intervals2 = []
    in_obscuration2 = False
    interval_start2 = None
    t = t_start
    while t <= t_end:
        m = missile_pos(t)
        c = cloud_center(t)
        if c is None:
            t += dt
            continue
        obscured = is_obscurated(m, c, target_pts)
        if obscured and not in_obscuration2:
            in_obscuration2 = True
            interval_start2 = t
        elif not obscured and in_obscuration2:
            in_obscuration2 = False
            obscuration_intervals2.append((interval_start2, t))
        t += dt
    if in_obscuration2:
        obscuration_intervals2.append((interval_start2, t_end))
    total_time2 = sum(b - a for a, b in obscuration_intervals2)

    print(f"\n--- 验证（代表点法）---")
    print(f"有效遮蔽时间区间:")
    for i, (a, b) in enumerate(obscuration_intervals2):
        print(f"  区间{i+1}: [{a:.3f}, {b:.3f}]s, 时长 {b-a:.3f}s")
    print(f"总有效遮蔽时长: {total_time2:.4f}s")

    # 打印关键时刻的状态
    print(f"\n--- 关键时刻状态 ---")
    for t_check in [T_DETON, T_DETON + 5, T_DETON + 10, T_DETON + 15, T_DETON + 20]:
        if t_check > t_end:
            break
        m = missile_pos(t_check)
        c = cloud_center(t_check)
        d_mc = np.linalg.norm(c - m)
        # 导弹到目标中心的距离
        d_mt = np.linalg.norm(TARGET_CENTER - m)
        # 云心到导弹-目标中心连线的距离
        dist_line, _ = point_to_segment_dist(c, m, TARGET_CENTER)
        print(f"t={t_check:.1f}s: M1=({m[0]:.1f},{m[1]:.1f},{m[2]:.1f}) "
              f"Cloud=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) "
              f"d(M,C)={d_mc:.1f} d(M,T)={d_mt:.1f} dist(C,LOS)={dist_line:.1f}")

    return total_time, obscuration_intervals

if __name__ == "__main__":
    compute_obscuration_time()
