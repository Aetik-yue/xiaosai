"""
FY2~FY5 密集网格搜索 - 找到各UAV的有效遮蔽参数
策略: dt_fuse=0(即时起爆), v=140(最大速度), 只搜theta和t_rel
"""
import numpy as np
from core import *
from optimizer import calc_time_fast, calc_time_precise, calc_multi_bomb_time

def grid_search_uav(uav_id, missile_id, theta_range, t_rel_range):
    """2维网格搜索: theta x t_rel, 固定v=140, dt_fuse=0"""
    uav0 = UAVS[uav_id]
    best_T = 0.0
    best_params = None

    for theta_deg in theta_range:
        theta = np.radians(theta_deg)
        for t_rel in t_rel_range:
            # dt_fuse=0, v=140
            T, _, _, _ = calc_time_fast(missile_id, uav_id, theta, 140.0,
                                         t_rel, 0.0, dt_step=0.05, n_samples=60)
            if T > best_T:
                best_T = T
                best_params = (theta, 140.0, t_rel, 0.0)

    return best_params, best_T


def grid_search_multi(uav_id, missile_id, theta_range, t_rel_range, n_bombs=2):
    """多弹网格搜索: 共享theta和v, 搜索各弹t_rel"""
    best_T = 0.0
    best_params = None

    for theta_deg in theta_range:
        theta = np.radians(theta_deg)
        for tr1 in t_rel_range:
            if n_bombs == 1:
                T, _, _, _ = calc_time_fast(missile_id, uav_id, theta, 140.0,
                                             tr1, 0.0, dt_step=0.05, n_samples=60)
                if T > best_T:
                    best_T = T
                    best_params = (theta, 140.0, [(tr1, 0.0)])
            elif n_bombs == 2:
                for tr2 in t_rel_range:
                    if tr2 < tr1 + 1:
                        continue
                    bombs = [(tr1, 0.0), (tr2, 0.0)]
                    T, _, _ = calc_multi_bomb_time(missile_id, uav_id, theta, 140.0,
                                                    bombs, dt_step=0.05, n_samples=60)
                    if T > best_T:
                        best_T = T
                        best_params = (theta, 140.0, bombs)
    return best_params, best_T


def solve_all():
    from optimizer import solve_q2

    # Q2
    print("="*60)
    print("Q2: 单弹优化")
    print("="*60)
    q2_params, q2_T, _, _ = solve_q2(n_particles=25, n_iter=30)

    # Q3 (复用v2结果)
    print("\n"+"="*60)
    print("Q3: 三弹接力(8变量PSO)")
    print("="*60)
    from solve_v2 import solve_q3_v2
    q3_theta, q3_v, q3_bombs, q3_T, q3_merged = solve_q3_v2(q2_params)

    # Q4: 网格搜索FY2, FY3
    print("\n"+"="*60)
    print("Q4: 三机协同(网格搜索)")
    print("="*60)
    q4_results = {'FY1': tuple(q2_params)}

    # FY2: 需要朝y负方向飞接近M1视线
    print("  FY2 网格搜索...")
    fy2_params, fy2_T = grid_search_uav('FY2', 'M1',
        theta_range=range(280, 370, 3), t_rel_range=np.arange(0, 30, 0.5))
    if fy2_params:
        # PSO精修
        theta0, v0, tr0, df0 = fy2_params
        rng = np.random.RandomState(42)
        n_p, n_i = 20, 25
        lb = np.array([theta0-0.3, 70, max(0, tr0-3), 0])
        ub = np.array([theta0+0.3, 140, tr0+3, 3])
        pos = rng.uniform(lb, ub, (n_p, 4))
        pos[0] = list(fy2_params)
        vel = np.zeros((n_p, 4))
        pbest = pos.copy()
        def fit(x, uid='FY2'):
            th, vv, tr, df = x
            if UAVS[uid][2]-0.5*G*df**2 < 0: return 0.0
            T, _, _, _ = calc_time_fast('M1', uid, th, vv, tr, df, dt_step=0.02, n_samples=100)
            return T
        pf = np.array([fit(p) for p in pos])
        gi = np.argmax(pf)
        gb, gf = pos[gi].copy(), pf[gi]
        for it in range(n_i):
            r1, r2 = rng.random((n_p,4)), rng.random((n_p,4))
            vel = 0.6*vel + 1.8*r1*(pbest-pos) + 1.8*r2*(gb-pos)
            pos = np.clip(pos+vel, lb, ub)
            for i in range(n_p):
                f = fit(pos[i])
                if f > pf[i]: pf[i], pbest[i] = f, pos[i].copy()
                if f > gf: gf, gb = f, pos[i].copy()
        q4_results['FY2'] = tuple(gb)
        T2_precise, _, _, _ = calc_time_precise('M1', 'FY2', *gb)
        print(f"  FY2: θ={np.degrees(gb[0]):.2f}°, v={gb[1]:.1f}, T={T2_precise:.4f}s")
    else:
        print("  FY2: 未找到有效参数")
        q4_results['FY2'] = (0.0, 100.0, 1.0, 0.0)

    # FY3: 类似
    print("  FY3 网格搜索...")
    fy3_params, fy3_T = grid_search_uav('FY3', 'M1',
        theta_range=range(0, 100, 3), t_rel_range=np.arange(0, 50, 0.5))
    if fy3_params:
        theta0, v0, tr0, df0 = fy3_params
        rng = np.random.RandomState(43)
        n_p, n_i = 20, 25
        lb = np.array([theta0-0.3, 70, max(0, tr0-3), 0])
        ub = np.array([theta0+0.3, 140, tr0+3, 3])
        pos = rng.uniform(lb, ub, (n_p, 4))
        pos[0] = list(fy3_params)
        vel = np.zeros((n_p, 4))
        pbest = pos.copy()
        def fit3(x, uid='FY3'):
            th, vv, tr, df = x
            if UAVS[uid][2]-0.5*G*df**2 < 0: return 0.0
            T, _, _, _ = calc_time_fast('M1', uid, th, vv, tr, df, dt_step=0.02, n_samples=100)
            return T
        pf = np.array([fit3(p) for p in pos])
        gi = np.argmax(pf)
        gb, gf = pos[gi].copy(), pf[gi]
        for it in range(n_i):
            r1, r2 = rng.random((n_p,4)), rng.random((n_p,4))
            vel = 0.6*vel + 1.8*r1*(pbest-pos) + 1.8*r2*(gb-pos)
            pos = np.clip(pos+vel, lb, ub)
            for i in range(n_p):
                f = fit3(pos[i])
                if f > pf[i]: pf[i], pbest[i] = f, pos[i].copy()
                if f > gf: gf, gb = f, pos[i].copy()
        q4_results['FY3'] = tuple(gb)
        T3_precise, _, _, _ = calc_time_precise('M1', 'FY3', *gb)
        print(f"  FY3: θ={np.degrees(gb[0]):.2f}°, v={gb[1]:.1f}, T={T3_precise:.4f}s")
    else:
        print("  FY3: 未找到有效参数")
        q4_results['FY3'] = (0.0, 100.0, 1.0, 0.0)

    # Q4并集
    all_iv = []
    for uid, (th, vv, tr, df) in q4_results.items():
        T, iv, _, _ = calc_time_precise('M1', uid, th, vv, tr, df)
        all_iv.extend(iv)
        print(f"  {uid}: T={T:.4f}s")
    all_iv.sort()
    merged = [list(all_iv[0])] if all_iv else []
    for s, e in all_iv[1:]:
        if s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([s, e])
    q4_T = sum(e-s for s,e in merged) if merged else 0
    print(f"\nQ4联合: {q4_T:.4f}s")

    # Q5: 5机多对多
    print("\n"+"="*60)
    print("Q5: 多对多(网格搜索)")
    print("="*60)
    assignment = {
        'FY1': ('M1', 2), 'FY2': ('M1', 1), 'FY3': ('M2', 2),
        'FY4': ('M3', 2), 'FY5': ('M3', 1),
    }
    q5_results = {}
    q5_T = 0.0

    for uid, (mid, nb) in assignment.items():
        print(f"  {uid}→{mid} ({nb}弹)...")
        # 确定搜索方向
        uav0 = UAVS[uid]
        m0 = MISSILES[mid]
        # 朝导弹→真目标视线方向
        base_theta = np.degrees(np.arctan2(-(uav0[1]), -(uav0[0]-m0[0]/2)))
        trange = range(int(base_theta-30)%360, int(base_theta+30)%360, 3)

        params, T = grid_search_multi(uid, mid, trange,
                                       np.arange(0, 40, 0.5), n_bombs=nb)
        if params:
            th, vv, bombs = params
            if nb == 1:
                T_p, iv, _, _ = calc_time_precise(mid, uid, th, vv, bombs[0][0], bombs[0][1])
            else:
                T_p, singles, merged5 = calc_multi_bomb_time(mid, uid, th, vv, bombs,
                                                              dt_step=0.01, n_samples=180)
            q5_results[uid] = (mid, [(th, vv, bombs, T_p, iv if nb==1 else merged5)])
            q5_T += T_p
            print(f"    T={T_p:.4f}s")
        else:
            print(f"    未找到有效参数")
            q5_results[uid] = (mid, [(0, 100, [(1, 0)], 0, [])])

    print(f"\nQ5总遮蔽: {q5_T:.4f}s")

    # 汇总
    print("\n"+"="*60)
    print("最终汇总")
    print("="*60)
    print(f"Q1: 1.3920s")
    print(f"Q2: {q2_T:.4f}s")
    print(f"Q3: {q3_T:.4f}s")
    print(f"Q4: {q4_T:.4f}s")
    print(f"Q5: {q5_T:.4f}s")

    return q2_params, q3_theta, q3_v, q3_bombs, q3_T, q4_results, q4_T, q5_results, q5_T


if __name__ == '__main__':
    solve_all()
