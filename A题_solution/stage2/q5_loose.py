"""
宽松判定版 Q5: 用A1的点到线段距离≤R重新计算
验证严格/宽松判定对多弹有效性的影响
"""
import numpy as np, json
from core import *

def log(msg): print(msg, flush=True)

# ===== 宽松判定函数(A1方法) =====
def is_occluded_loose(missile_id, cloud_pos, t):
    """A1宽松判定: 云团球心到导弹→真目标中心线段的距离≤R"""
    m_pos = missile_pos(missile_id, t)
    target = TARGET_CENTER  # (0,200,0)
    # 线段M→A参数化: L(u) = u*M + (1-u)*A, u∈[0,1]
    # 线段L(u) = M + u*(A-M) = M + u*(-A+M的反方向)...
    # 实际上: L(u) = (1-u)*M + u*A, u∈[0,1] 或 L(u) = M + s*(A-M), s∈[0,1]
    MA = target - m_pos  # 从M到A的向量
    MC = cloud_pos - m_pos  # 从M到云团中心的向量
    u = np.dot(MC, MA) / np.dot(MA, MA) if np.dot(MA, MA) > 1e-10 else 0.5
    u = np.clip(u, 0, 1)
    closest = m_pos + u * MA
    d = np.linalg.norm(cloud_pos - closest)
    return d <= R_SMOKE

def calc_loose_time(missile_id, uav_id, theta, v, t_rel, dt_fuse,
                    dt_step=0.01, t_limit=None):
    """宽松判定遮蔽时长"""
    burst_pos, t_burst = burst_point(uav_id, theta, v, t_rel, dt_fuse)
    if burst_pos[2] < 0:
        return 0.0, [], burst_pos, t_burst
    if t_limit is None:
        t_limit = t_burst + 20
    t_end = min(t_burst + 20, t_limit)
    total = 0.0
    in_occ, i_start = False, None
    intervals = []
    n = int((t_end - t_burst) / dt_step)
    for i in range(n):
        t = t_burst + i*dt_step
        c = burst_pos.copy()
        c[2] -= 3*(t - t_burst)
        if is_occluded_loose(missile_id, c, t):
            if not in_occ:
                in_occ, i_start = True, t
        else:
            if in_occ:
                in_occ = False
                intervals.append((i_start, t))
                total += t - i_start
    if in_occ:
        intervals.append((i_start, t_end))
        total += t_end - i_start
    return total, intervals, burst_pos, t_burst

def calc_multi_loose(missile_id, uav_id, theta, v, bombs, dt_step=0.01):
    """宽松判定多弹并集"""
    all_iv = []
    singles = []
    for tr, df in bombs:
        T, iv, _, _ = calc_loose_time(missile_id, uav_id, theta, v, tr, df, dt_step)
        all_iv.extend(iv)
        singles.append(T)
    if not all_iv:
        return 0.0, singles, []
    all_iv.sort()
    merged = [list(all_iv[0])]
    for s,e in all_iv[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s,e])
    union = sum(e-s for s,e in merged)
    return union, singles, merged

# ===== 主程序 =====
with open(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\q5_updated.json") as f:
    old = json.load(f)

log("="*60)
log("Q5 宽松判定 (A1方法): 点到线段距离≤R")
log("="*60)

assign = {uid: old['details'][uid]['missile'] for uid in old['details']}
# 用之前严格判定的最优参数
total_loose = 0.0

for uid in ['FY1','FY2','FY3','FY4','FY5']:
    mid = assign[uid]
    d = old['details'][uid]
    th = np.radians(d['theta_deg'])
    v = d['v']
    tr1 = d['t_rel']
    df1 = d['dt_fuse']

    log(f"\n--- {uid}→{mid} (θ={d['theta_deg']:.2f}°, v={v:.1f}) ---")

    # 弹1宽松判定
    T1, iv1, _, _ = calc_loose_time(mid, uid, th, v, tr1, df1, dt_step=0.005)
    log(f"  弹1(宽松): T={T1:.4f}s, 区间={[(f'{s:.2f}',f'{e:.2f}') for s,e in iv1]}")

    # 搜索第2弹
    best_u2, best_b2 = T1, None
    for df2 in np.arange(0, 10.5, 0.5):
        if UAVS[uid][2] - 4.9*df2**2 < 0: break
        for tr2 in np.arange(tr1+1, 50, 0.5):
            Tu, singles, _ = calc_multi_loose(mid, uid, th, v, [(tr1,df1),(tr2,df2)])
            if Tu > best_u2 + 0.05:
                best_u2 = Tu
                best_b2 = (tr2, df2, Tu)

    if best_b2 and best_u2 > T1 + 0.1:
        tr2, df2, Tu2 = best_b2
        log(f"  弹2: tr={tr2:.1f}, df={df2:.1f}, 并集={Tu2:.4f}s (增量={Tu2-T1:.3f}s)")
        all_bombs = [(tr1,df1),(tr2,df2)]
        T_current = Tu2

        # 搜索第3弹
        best_u3, best_b3 = T_current, None
        for df3 in np.arange(0, 10.5, 0.5):
            if UAVS[uid][2] - 4.9*df3**2 < 0: break
            for tr3 in np.arange(tr2+1, 55, 0.5):
                Tu, _, _ = calc_multi_loose(mid, uid, th, v, all_bombs+[(tr3,df3)])
                if Tu > best_u3 + 0.05:
                    best_u3 = Tu
                    best_b3 = (tr3, df3, Tu)

        if best_b3 and best_u3 > T_current + 0.1:
            tr3, df3, Tu3 = best_b3
            log(f"  弹3: tr={tr3:.1f}, df={df3:.1f}, 并集={Tu3:.4f}s (增量={Tu3-T_current:.3f}s)")
            T_current = Tu3
    else:
        log(f"  第2弹无增量")
        T_current = T1

    total_loose += T_current
    log(f"  → 宽松判定并集: {T_current:.4f}s")

log(f"\n{'='*60}")
log(f"宽松判定Q5总计: {total_loose:.4f}s")
log(f"严格判定Q5总计: 18.920s")
log(f"{'='*60}")
