"""
A题烟幕干扰弹 - 向量化核心 + 优化器
Q2: PSO单弹优化
Q3: 分层+DEGA多弹接力
"""
import numpy as np
from core import *

# ============ 向量化遮蔽判定(加速) ============

def _sample_points(n_samples):
    """预生成圆柱上下圆周采样点"""
    phi = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    # 下底面
    bot = np.stack([TARGET_RADIUS * cos_p, TARGET_RADIUS * sin_p,
                    np.zeros(n_samples)], axis=1)
    bot[:, 0] += TARGET_CENTER[0]
    bot[:, 1] += TARGET_CENTER[1]
    # 上底面
    top = bot.copy()
    top[:, 2] += TARGET_HEIGHT
    return np.vstack([bot, top])  # (2*n_samples, 3)


_SAMPLE_CACHE = {}
def get_samples(n_samples):
    if n_samples not in _SAMPLE_CACHE:
        _SAMPLE_CACHE[n_samples] = _sample_points(n_samples)
    return _SAMPLE_CACHE[n_samples]


def is_occluded_vec(missile_id, cloud_pos, t, n_samples=180):
    """向量化遮蔽判定"""
    m_pos = missile_pos(missile_id, t)
    mc = cloud_pos - m_pos
    d = np.linalg.norm(mc)
    if d <= R_SMOKE:
        return True
    alpha = np.arcsin(R_SMOKE / d)

    pts = get_samples(n_samples)
    mp = pts - m_pos  # (2n, 3)
    mp_norm = np.linalg.norm(mp, axis=1)  # (2n,)
    mc_norm = d
    cos_beta = (mp @ mc) / (mp_norm * mc_norm + 1e-30)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)
    beta_max = beta.max()
    return beta_max <= alpha


def calc_time_fast(missile_id, uav_id, theta, v, t_rel, dt_fuse,
                   dt_step=0.005, n_samples=180):
    """快速遮蔽时长(优化器内循环用)"""
    burst_pos, t_burst = burst_point(uav_id, theta, v, t_rel, dt_fuse)
    if burst_pos[2] < 0:
        return 0.0, [], burst_pos, t_burst

    t_start = t_burst
    t_end = t_burst + EFFECTIVE_WINDOW
    total = 0.0
    intervals = []
    in_occ = False
    i_start = None

    n_steps = int((t_end - t_start) / dt_step)
    for i in range(n_steps):
        t = t_start + i * dt_step
        c = burst_pos.copy()
        c[2] -= V_SINK * (t - t_burst)
        if is_occluded_vec(missile_id, c, t, n_samples):
            if not in_occ:
                in_occ = True
                i_start = t
        else:
            if in_occ:
                in_occ = False
                intervals.append((i_start, t))
                total += t - i_start
    if in_occ:
        intervals.append((i_start, t_end))
        total += t_end - i_start
    return total, intervals, burst_pos, t_burst


def calc_time_precise(missile_id, uav_id, theta, v, t_rel, dt_fuse,
                      dt_step=0.001, n_samples=360):
    """精确遮蔽时长(最终结果用)"""
    return calc_time_fast(missile_id, uav_id, theta, v, t_rel, dt_fuse,
                          dt_step, n_samples)


# ============ 多弹并集遮蔽时长 ============

def calc_multi_bomb_time(missile_id, uav_id, theta, v, bombs,
                         dt_step=0.005, n_samples=180):
    """多弹并集遮蔽时长
    bombs: [(t_rel_j, dt_fuse_j), ...]
    返回: (并集时长, 各弹单独时长列表, 各弹区间列表)
    """
    all_intervals = []
    single_times = []
    for t_rel, dt_fuse in bombs:
        T, intervals, _, _ = calc_time_fast(missile_id, uav_id, theta, v,
                                             t_rel, dt_fuse, dt_step, n_samples)
        all_intervals.extend(intervals)
        single_times.append(T)

    # 并集
    if not all_intervals:
        return 0.0, single_times, []
    all_intervals.sort()
    merged = [list(all_intervals[0])]
    for s, e in all_intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    union_time = sum(e - s for s, e in merged)
    return union_time, single_times, merged


# ============ Q2: PSO单弹优化 ============

def solve_q2(n_particles=30, n_iter=40, seed=42):
    """Q2: 优化FY1单弹对M1的遮蔽时长
    变量: theta(0~2pi), v(70~140), t_rel(0~10), dt_fuse(0~6)
    """
    rng = np.random.RandomState(seed)
    lb = np.array([0.0, 70.0, 0.0, 0.0])
    ub = np.array([2 * np.pi, 140.0, 10.0, 6.0])
    dim = 4

    # 初始化粒子
    pos = rng.uniform(lb, ub, (n_particles, dim))
    # 给一个朝假目标方向的初始解(加速收敛)
    pos[0] = [np.pi, 120.0, 1.5, 3.6]
    vel = np.zeros((n_particles, dim))

    def fitness(x):
        theta, v, t_rel, dt_fuse = x
        if v < 70 or v > 140:
            return 0.0
        burst_z = UAVS['FY1'][2] - 0.5 * G * dt_fuse ** 2
        if burst_z < 0:
            return 0.0
        T, _, _, _ = calc_time_fast('M1', 'FY1', theta, v, t_rel, dt_fuse,
                                     dt_step=0.01, n_samples=120)
        return T

    pbest_pos = pos.copy()
    pbest_fit = np.array([fitness(p) for p in pos])
    gbest_idx = np.argmax(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    w, c1, c2 = 0.7, 1.5, 1.5
    for it in range(n_iter):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = pos + vel
        pos = np.clip(pos, lb, ub)

        for i in range(n_particles):
            f = fitness(pos[i])
            if f > pbest_fit[i]:
                pbest_fit[i] = f
                pbest_pos[i] = pos[i].copy()
            if f > gbest_fit:
                gbest_fit = f
                gbest_pos = pos[i].copy()

        if (it + 1) % 10 == 0:
            print(f"  PSO iter {it+1}/{n_iter}: gbest={gbest_fit:.4f}s")

    # 精确验证
    theta, v, t_rel, dt_fuse = gbest_pos
    T_precise, intervals, burst_pos, t_burst = calc_time_precise(
        'M1', 'FY1', theta, v, t_rel, dt_fuse)

    print(f"\nQ2结果(精确):")
    print(f"  航向角: {np.degrees(theta):.2f}°")
    print(f"  速度: {v:.2f} m/s")
    print(f"  投放时刻: {t_rel:.4f}s")
    print(f"  引信延迟: {dt_fuse:.4f}s")
    print(f"  起爆点: ({burst_pos[0]:.2f}, {burst_pos[1]:.2f}, {burst_pos[2]:.2f})")
    print(f"  有效遮蔽时长: {T_precise:.4f}s")
    print(f"  遮蔽区间: {[(f'{s:.3f}', f'{e:.3f}') for s, e in intervals]}")
    return gbest_pos, T_precise, intervals, burst_pos


if __name__ == '__main__':
    print("=" * 60)
    print("Q2: FY1单弹优化")
    print("=" * 60)
    solve_q2(n_particles=30, n_iter=40)
