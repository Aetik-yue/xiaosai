"""
A题烟幕干扰弹投放策略 - 核心模型
模型一:运动学(导弹/无人机/烟幕弹/云团)
模型二:遮蔽判定(严格视锥)
"""
import numpy as np
from scipy.optimize import minimize_scalar

G = 9.8  # 重力加速度
R_SMOKE = 10.0  # 烟幕云团半径(m)
V_MISSILE = 300.0  # 导弹速度(m/s)
V_SINK = 3.0  # 云团下沉速度(m/s)
EFFECTIVE_WINDOW = 20.0  # 云团有效时间(s)

# 真目标圆柱参数
TARGET_CENTER = np.array([0.0, 200.0, 0.0])  # 下底面圆心
TARGET_RADIUS = 7.0  # 圆柱半径
TARGET_HEIGHT = 10.0  # 圆柱高度

# 假目标(原点)
FAKE_TARGET = np.array([0.0, 0.0, 0.0])

# 导弹初始位置
MISSILES = {
    'M1': np.array([20000.0, 0.0, 2000.0]),
    'M2': np.array([19000.0, 600.0, 2100.0]),
    'M3': np.array([18000.0, -600.0, 1900.0]),
}

# 无人机初始位置
UAVS = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0]),
}


# ============ 模型一:运动学 ============

def missile_pos(missile_id, t):
    """导弹在时刻t的位置(匀速直线指向原点)"""
    m0 = MISSILES[missile_id]
    direction = (FAKE_TARGET - m0) / np.linalg.norm(FAKE_TARGET - m0)
    return m0 + V_MISSILE * t * direction


def missile_hit_time(missile_id):
    """导弹命中原点时刻"""
    m0 = MISSILES[missile_id]
    return np.linalg.norm(m0) / V_MISSILE


def uav_pos(uav_id, theta, v, t):
    """无人机在时刻t的位置(水平匀速直线)
    theta: 航向角(弧度,以x轴正向逆时针)
    v: 速度(m/s)
    """
    u0 = UAVS[uav_id]
    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    return u0 + v * t * direction


def burst_point(uav_id, theta, v, t_rel, dt_fuse):
    """烟幕弹起爆点坐标
    t_rel: 投放时刻(从受领任务起)
    dt_fuse: 引信延迟(投放到起爆)
    """
    t_burst = t_rel + dt_fuse
    # 投放点 = 无人机在t_rel的位置
    drop_pos = uav_pos(uav_id, theta, v, t_rel)
    # 起爆点 = 投放点 + 水平惯性v*dt_fuse - 垂直下落
    direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    burst = drop_pos + v * dt_fuse * direction
    burst[2] -= 0.5 * G * dt_fuse ** 2
    return burst, t_burst


def cloud_center(burst_pos, t_burst, t):
    """烟幕云团球心在时刻t的位置
    起爆后水平固定,以3m/s匀速下沉
    """
    if t < t_burst:
        return None
    c = burst_pos.copy()
    c[2] -= V_SINK * (t - t_burst)
    return c


# ============ 模型二:遮蔽判定(严格视锥) ============

def occlusion_angle(missile_id, cloud_pos, t):
    """遮蔽半顶角 alpha = arcsin(R/d)
    d: 导弹到云团球心距离
    """
    m_pos = missile_pos(missile_id, t)
    d = np.linalg.norm(m_pos - cloud_pos)
    if d <= R_SMOKE:
        return np.pi / 2  # 导弹在云团内,直接遮蔽
    return np.arcsin(R_SMOKE / d)


def target_max_angle(missile_id, cloud_pos, t, n_samples=360):
    """目标圆柱最大视角 beta_max
    对上下底面圆周采样,求最大beta
    """
    m_pos = missile_pos(missile_id, t)
    mc = cloud_pos - m_pos
    mc_norm = np.linalg.norm(mc)

    if mc_norm < 1e-10:
        return 0.0

    # 下底面圆心(0,200,0), 上底面圆心(0,200,10)
    bot_center = TARGET_CENTER.copy()
    top_center = TARGET_CENTER.copy()
    top_center[2] += TARGET_HEIGHT

    beta_max = 0.0
    # 上下圆周各采样n_samples个点
    for phi in np.linspace(0, 2 * np.pi, n_samples, endpoint=False):
        offset = TARGET_RADIUS * np.array([np.cos(phi), np.sin(phi), 0.0])
        for center in [bot_center, top_center]:
            p = center + offset
            mp = p - m_pos
            mp_norm = np.linalg.norm(mp)
            if mp_norm < 1e-10:
                continue
            cos_beta = np.dot(mp, mc) / (mp_norm * mc_norm)
            cos_beta = np.clip(cos_beta, -1.0, 1.0)
            beta = np.arccos(cos_beta)
            if beta > beta_max:
                beta_max = beta
    return beta_max


def is_occluded(missile_id, cloud_pos, t, n_samples=360):
    """判断时刻t是否有效遮蔽
    判据: 导弹在云团内(d<=R)直接遮蔽; 否则 beta_max <= alpha
    """
    m_pos = missile_pos(missile_id, t)
    d = np.linalg.norm(m_pos - cloud_pos)
    if d <= R_SMOKE:
        return True  # 导弹在云团内,完全包围,遮蔽有效
    alpha = np.arcsin(R_SMOKE / d)
    beta_max = target_max_angle(missile_id, cloud_pos, t, n_samples)
    return beta_max <= alpha


def calc_occlusion_time(missile_id, uav_id, theta, v, t_rel, dt_fuse,
                        dt_step=0.001, n_samples=360):
    """计算单弹有效遮蔽时长
    返回: (遮蔽时长, 遮蔽区间列表, 起爆点, 起爆时刻)
    """
    burst_pos, t_burst = burst_point(uav_id, theta, v, t_rel, dt_fuse)

    # 检查起爆点不穿地
    if burst_pos[2] < 0:
        return 0.0, [], burst_pos, t_burst

    t_start = t_burst
    t_end = t_burst + EFFECTIVE_WINDOW
    total_time = 0.0
    intervals = []
    in_occlusion = False
    interval_start = None

    n_steps = int((t_end - t_start) / dt_step)
    for i in range(n_steps):
        t = t_start + i * dt_step
        cloud_pos = cloud_center(burst_pos, t_burst, t)
        if cloud_pos is None:
            continue
        occluded = is_occluded(missile_id, cloud_pos, t, n_samples)
        if occluded and not in_occlusion:
            in_occlusion = True
            interval_start = t
        elif not occluded and in_occlusion:
            in_occlusion = False
            intervals.append((interval_start, t))
            total_time += t - interval_start

    if in_occlusion:
        intervals.append((interval_start, t_end))
        total_time += t_end - interval_start

    return total_time, intervals, burst_pos, t_burst


def calc_occlusion_time_fast(missile_id, uav_id, theta, v, t_rel, dt_fuse,
                             dt_step=0.001, n_samples=180):
    """快速版遮蔽时长计算(用于优化器内循环,减少采样点)"""
    return calc_occlusion_time(missile_id, uav_id, theta, v, t_rel, dt_fuse,
                               dt_step, n_samples)


# ============ Q1: 验证 ============

def solve_q1():
    """Q1: FY1以120m/s朝假目标飞行,1.5s后投放,3.6s后起爆
    求对M1的有效遮蔽时长
    """
    # FY1朝假目标(原点)飞行 -> theta = atan2(0-0, 0-17800) = pi
    uav0 = UAVS['FY1']
    theta = np.arctan2(FAKE_TARGET[1] - uav0[1], FAKE_TARGET[0] - uav0[0])
    v = 120.0
    t_rel = 1.5
    dt_fuse = 3.6

    print(f"Q1参数:")
    print(f"  FY1朝向角: {np.degrees(theta):.2f}° (朝假目标)")
    print(f"  速度: {v} m/s")
    print(f"  投放时刻: {t_rel}s")
    print(f"  引信延迟: {dt_fuse}s")

    burst_pos, t_burst = burst_point('FY1', theta, v, t_rel, dt_fuse)
    print(f"  起爆点: ({burst_pos[0]:.2f}, {burst_pos[1]:.2f}, {burst_pos[2]:.2f})")
    print(f"  起爆时刻: {t_burst:.2f}s")

    T, intervals, _, _ = calc_occlusion_time('M1', 'FY1', theta, v, t_rel, dt_fuse,
                                              dt_step=0.001, n_samples=360)
    print(f"\nQ1结果:")
    print(f"  有效遮蔽时长: {T:.4f}s")
    print(f"  遮蔽区间: {[(f'{s:.3f}', f'{e:.3f}') for s, e in intervals]}")
    return T, intervals, burst_pos, t_burst


if __name__ == '__main__':
    solve_q1()
