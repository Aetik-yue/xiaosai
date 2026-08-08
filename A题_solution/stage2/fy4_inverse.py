"""
FY4 逆向运动学搜索: 直接求解FY4到达视线的可行性条件
不再暴力网格,而是从"遮蔽时刻t + 视线参数u"反推UAV参数
"""
import numpy as np
from core import *
from optimizer import calc_time_fast, calc_time_precise

def log(msg): print(msg, flush=True)

def inverse_kinematics_search(uav_id, missile_id):
    """逆向运动学: 对每个(t, u, dt_fuse)求解UAV参数"""
    uav0 = UAVS[uav_id]
    best_T, best_params = 0.0, None

    for dt_fuse in np.arange(0, 10, 0.5):
        burst_z = uav0[2] - 4.9 * dt_fuse ** 2
        if burst_z < 0:
            continue
        for t in np.arange(3, 60, 0.5):
            m_pos = missile_pos(missile_id, t)
            # 视线参数化: L(u) = u*TARGET + (1-u)*M(t)
            # C = L(u) = ((1-u)*Mx, 200u, (1-u)*Mz)  (真目标(0,200,0))
            # 但真目标中心是(0,200,0), 视线到中心
            # 实际视线是 M(t) → 真目标圆柱表面, 用中心近似
            target = TARGET_CENTER  # (0,200,0)

            for u in np.arange(0.01, 0.99, 0.05):
                # 云团球心 = 视线上的点
                Cx = (1-u) * m_pos[0] + u * target[0]
                Cy = (1-u) * m_pos[1] + u * target[1]
                Cz = (1-u) * m_pos[2] + u * target[2]

                # 起爆点z = Cz + 3*(t - t_b), t_b = t_rel + dt_fuse
                # 起爆点z = uav_z - 4.9*dt_fuse^2
                # 所以: Cz + 3*(t - t_b) = uav_z - 4.9*dt_fuse^2
                # t_b = (Cz + 3*t - uav_z + 4.9*dt_fuse^2) / 3
                t_b = (Cz + 3*t - uav0[2] + 4.9*dt_fuse**2) / 3
                if t_b <= 0:
                    continue
                t_rel = t_b - dt_fuse
                if t_rel < 0:
                    continue

                # 起爆点x,y = UAV位置 + v*t_b*方向
                # 起爆点x = uav_x + v*t_b*cos(theta) = Cx
                # 起爆点y = uav_y + v*t_b*sin(theta) = Cy
                dx = Cx - uav0[0]
                dy = Cy - uav0[1]
                dist_xy = np.sqrt(dx**2 + dy**2)

                # v*t_b = dist_xy → v = dist_xy / t_b
                v = dist_xy / t_b
                if v < 70 or v > 140:
                    continue

                theta = np.arctan2(dy, dx)

                # 验证: 用求得的参数计算遮蔽时长
                T, _, _, _ = calc_time_fast(missile_id, uav_id, theta, v, t_rel, dt_fuse,
                                             dt_step=0.05, n_samples=60)
                if T > best_T:
                    best_T = T
                    best_params = (theta, v, t_rel, dt_fuse)
                    if T > 0.5:
                        log(f"    找到! t={t:.1f}, u={u:.2f}, df={dt_fuse:.1f}, "
                            f"θ={np.degrees(theta):.1f}°, v={v:.1f}, tr={t_rel:.2f}, T={T:.3f}s")

    return best_params, best_T


def pso_refine(uav_id, missile_id, init_p, n_p=30, n_i=40):
    th0, v0, tr0, df0 = init_p
    rng = np.random.RandomState(hash(uav_id+missile_id+"refine")%1000)
    lb = np.array([th0-0.5, max(70,v0-20), max(0,tr0-3), max(0,df0-2)])
    ub = np.array([th0+0.5, min(140,v0+20), tr0+3, df0+2])
    pos = rng.uniform(lb, ub, (n_p, 4))
    pos[0] = list(init_p)
    vel = np.zeros((n_p, 4))
    pbest = pos.copy()
    def fit(x):
        th, vv, tr, df = x
        if UAVS[uav_id][2]-0.5*G*df**2 < 0: return 0.0
        if vv < 70 or vv > 140: return 0.0
        T, _, _, _ = calc_time_fast(missile_id, uav_id, th, vv, tr, df, dt_step=0.02, n_samples=100)
        return T
    pf = np.array([fit(p) for p in pos])
    gi = np.argmax(pf); gb, gf = pos[gi].copy(), pf[gi]
    for it in range(n_i):
        r1, r2 = rng.random((n_p,4)), rng.random((n_p,4))
        vel = 0.6*vel + 1.8*r1*(pbest-pos) + 1.8*r2*(gb-pos)
        pos = np.clip(pos+vel, lb, ub)
        for i in range(n_p):
            f = fit(pos[i])
            if f > pf[i]: pf[i], pbest[i] = f, pos[i].copy()
            if f > gf: gf, gb = f, pos[i].copy()
    return tuple(gb), gf


if __name__ == '__main__':
    log("="*60)
    log("FY4 逆向运动学搜索")
    log("="*60)
    log(f"FY4初始位置: {UAVS['FY4']}")

    for mid in ['M1', 'M2', 'M3']:
        m0 = MISSILES[mid]
        log(f"\n--- FY4 → {mid} (导弹初始{m0}) ---")
        p, T = inverse_kinematics_search('FY4', mid)
        if p and T > 0.01:
            log(f"  逆向搜索最优: T={T:.4f}s")
            log(f"    θ={np.degrees(p[0]):.2f}°, v={p[1]:.2f}, tr={p[2]:.3f}, df={p[3]:.3f}")
            log(f"  PSO精修...")
            p2, T2 = pso_refine('FY4', mid, p)
            Tp, iv, bp, tb = calc_time_precise(mid, 'FY4', *p2)
            log(f"  精确验证: T={Tp:.4f}s")
            log(f"    θ={np.degrees(p2[0]):.2f}°, v={p2[1]:.2f}, tr={p2[2]:.3f}, df={p2[3]:.3f}")
            log(f"    起爆点: ({bp[0]:.1f}, {bp[1]:.1f}, {bp[2]:.1f})")
            log(f"    遮蔽区间: {[(f'{s:.3f}',f'{e:.3f}') for s,e in iv]}")
        else:
            log(f"  未找到有效遮蔽参数")
