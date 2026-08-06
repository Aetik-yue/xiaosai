"""
A题 Q3~Q5 修复版 - 智能初始解 + 全变量优化
"""
import numpy as np
import openpyxl
from core import *
from optimizer import calc_time_fast, calc_time_precise, calc_multi_bomb_time

# ============ 智能初始解生成 ============

def smart_init(uav_id, missile_id):
    """为指定UAV生成针对指定导弹的智能初始解
    思路: 找到导弹最接近UAV+真目标连线的时刻,在该时刻附近投弹
    """
    uav0 = UAVS[uav_id]
    m0 = MISSILES[missile_id]
    t_hit = np.linalg.norm(m0) / V_MISSILE

    # 扫描导弹轨迹,找UAV能遮蔽的最佳时刻
    best_T = 0.0
    best_params = None

    for t_target in np.arange(2, min(t_hit, 50), 2.0):
        m_pos = missile_pos(missile_id, t_target)
        # UAV需要朝导弹方向飞,让弹落在导弹→真目标视线附近
        # 简化: 让起爆点在导弹当前位置和真目标之间
        # 起爆点 = UAV位置 + v*t_burst*方向 - 下落
        # 需要起爆点在导弹→真目标连线附近

        # 视线中点
        mid_point = (m_pos + TARGET_CENTER) / 2

        # UAV到视线中点的方向
        dir_to_mid = mid_point - uav0
        dir_to_mid[2] = 0  # 水平方向
        if np.linalg.norm(dir_to_mid[:2]) < 1:
            continue
        theta = np.arctan2(dir_to_mid[1], dir_to_mid[0])

        # 尝试不同速度和时序
        for v in [70, 100, 140]:
            for t_rel in [0.5, 2.0, 5.0]:
                for dt_fuse in [0.5, 2.0, 4.0]:
                    burst_z = uav0[2] - 0.5 * G * dt_fuse ** 2
                    if burst_z < 0:
                        continue
                    T, _, _, _ = calc_time_fast(missile_id, uav_id, theta, v,
                                                 t_rel, dt_fuse,
                                                 dt_step=0.02, n_samples=100)
                    if T > best_T:
                        best_T = T
                        best_params = (theta, v, t_rel, dt_fuse)

    if best_params is None:
        best_params = (0.0, 100.0, 1.0, 2.0)
    return best_params, best_T


# ============ Q3: 8变量全优化 ============

def solve_q3_v2(q2_params):
    """Q3: FY1投3弹,8变量全优化(θ,v可变)"""
    print("Q3(v2): FY1三弹8变量优化")
    theta0, v0 = q2_params[0], q2_params[1]
    t_rel1, dt_fuse1 = q2_params[2], q2_params[3]

    rng = np.random.RandomState(123)
    n_p, n_i = 30, 40

    # 8变量: theta, v, tr1, df1, tr2, df2, tr3, df3
    lb = np.array([0.0, 70.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0])
    ub = np.array([2*np.pi, 140.0, 10.0, 6.0, 20.0, 6.0, 30.0, 6.0])
    dim = 8

    pos = rng.uniform(lb, ub, (n_p, dim))
    # 初始解1: Q2最优
    pos[0] = [theta0, v0, t_rel1, dt_fuse1, t_rel1+2, 2.0, t_rel1+5, 3.0]
    # 初始解2: 不同航向
    pos[1] = [0.1, 100, 1.0, 1.0, 5.0, 2.0, 10.0, 3.0]
    vel = np.zeros((n_p, dim))
    pbest = pos.copy()

    def fitness(x):
        th, vv, tr1, df1, tr2, df2, tr3, df3 = x
        if tr2 < tr1 + 1 or tr3 < tr2 + 1:
            return 0.0
        if UAVS['FY1'][2] - 0.5*G*df1**2 < 0 or UAVS['FY1'][2] - 0.5*G*df2**2 < 0:
            return 0.0
        if UAVS['FY1'][2] - 0.5*G*df3**2 < 0:
            return 0.0
        bombs = [(tr1, df1), (tr2, df2), (tr3, df3)]
        T, _, _ = calc_multi_bomb_time('M1', 'FY1', th, vv, bombs,
                                        dt_step=0.02, n_samples=100)
        return T

    pf = np.array([fitness(p) for p in pos])
    gi = np.argmax(pf)
    gb, gf = pos[gi].copy(), pf[gi]

    w, c1, c2 = 0.6, 1.8, 1.8
    for it in range(n_i):
        r1, r2 = rng.random((n_p, dim)), rng.random((n_p, dim))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gb - pos)
        pos = np.clip(pos + vel, lb, ub)
        for i in range(n_p):
            f = fitness(pos[i])
            if f > pf[i]:
                pf[i], pbest[i] = f, pos[i].copy()
            if f > gf:
                gf, gb = f, pos[i].copy()
        if (it+1) % 10 == 0:
            print(f"  Q3 iter {it+1}: gbest={gf:.4f}s")

    th, vv, tr1, df1, tr2, df2, tr3, df3 = gb
    bombs = [(tr1, df1), (tr2, df2), (tr3, df3)]
    T, singles, merged = calc_multi_bomb_time('M1', 'FY1', th, vv, bombs,
                                               dt_step=0.005, n_samples=240)
    print(f"\nQ3结果(精确):")
    print(f"  θ={np.degrees(th):.2f}°, v={vv:.1f}")
    for i, (tr, df) in enumerate(bombs):
        print(f"  弹{i+1}: t_rel={tr:.4f}, dt={df:.4f}, 单独={singles[i]:.4f}s")
    print(f"  并集: {T:.4f}s")
    return th, vv, bombs, T, merged


# ============ Q4: 智能初始解+PSO ============

def solve_q4_v2(q2_params):
    """Q4: 3机各1弹,智能初始解"""
    print("Q4(v2): 三机协同(智能初始解)")

    results = {'FY1': tuple(q2_params)}

    for uav_id in ['FY2', 'FY3']:
        print(f"  {uav_id} 智能初始解搜索...")
        init_params, init_T = smart_init(uav_id, 'M1')
        print(f"    初始解: θ={np.degrees(init_params[0]):.1f}°, v={init_params[1]:.1f}, T={init_T:.3f}s")

        rng = np.random.RandomState(hash(uav_id) % 1000)
        n_p, n_i = 25, 35
        lb = np.array([0.0, 70.0, 0.0, 0.0])
        ub = np.array([2*np.pi, 140.0, 15.0, 6.0])

        pos = rng.uniform(lb, ub, (n_p, 4))
        pos[0] = list(init_params)
        # 在初始解附近扰动
        for i in range(1, min(10, n_p)):
            pos[i] = np.clip(init_params + rng.normal(0, [0.3, 15, 2, 1]), lb, ub)
        vel = np.zeros((n_p, 4))
        pbest = pos.copy()

        def fit(x, uid=uav_id):
            th, vv, tr, df = x
            if UAVS[uid][2] - 0.5*G*df**2 < 0:
                return 0.0
            T, _, _, _ = calc_time_fast('M1', uid, th, vv, tr, df,
                                         dt_step=0.02, n_samples=100)
            return T

        pf = np.array([fit(p) for p in pos])
        gi = np.argmax(pf)
        gb, gf = pos[gi].copy(), pf[gi]

        for it in range(n_i):
            r1, r2 = rng.random((n_p, 4)), rng.random((n_p, 4))
            vel = 0.6*vel + 1.8*r1*(pbest-pos) + 1.8*r2*(gb-pos)
            pos = np.clip(pos + vel, lb, ub)
            for i in range(n_p):
                f = fit(pos[i])
                if f > pf[i]:
                    pf[i], pbest[i] = f, pos[i].copy()
                if f > gf:
                    gf, gb = f, pos[i].copy()
            if (it+1) % 10 == 0:
                print(f"    {uav_id} iter {it+1}: gbest={gf:.4f}s")

        results[uav_id] = tuple(gb)
        print(f"    {uav_id} 最终: θ={np.degrees(gb[0]):.2f}°, v={gb[1]:.1f}, T={gf:.4f}s")

    # 精确计算并集
    all_intervals = []
    for uav_id, (th, vv, tr, df) in results.items():
        T, intervals, _, _ = calc_time_precise('M1', uav_id, th, vv, tr, df)
        all_intervals.extend(intervals)
        print(f"  {uav_id}: T={T:.4f}s, 区间={[(f'{s:.2f}',f'{e:.2f}') for s,e in intervals]}")

    all_intervals.sort()
    merged = [list(all_intervals[0])] if all_intervals else []
    for s, e in all_intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    union_T = sum(e - s for s, e in merged) if merged else 0.0
    print(f"\nQ4结果: 联合={union_T:.4f}s, 区间数={len(merged)}")
    return results, union_T, merged


# ============ Q5: 智能分配+分组优化 ============

def solve_q5_v2(q4_results):
    """Q5: 5机多对多,智能初始解"""
    print("Q5(v2): 5机多对多协同")

    # 任务分配: 根据UAV与各导弹的距离/方位
    assignment = {
        'FY1': ('M1', 2),
        'FY2': ('M1', 1),
        'FY3': ('M2', 2),
        'FY4': ('M3', 2),
        'FY5': ('M3', 1),
    }

    all_results = {}
    total_T = 0.0

    for uav_id, (missile_id, n_bombs) in assignment.items():
        print(f"\n  {uav_id}→{missile_id} ({n_bombs}弹):")

        # 智能初始解
        init_params, init_T = smart_init(uav_id, missile_id)
        print(f"    初始解: θ={np.degrees(init_params[0]):.1f}°, v={init_params[1]:.1f}, T={init_T:.3f}s")

        rng = np.random.RandomState(hash(uav_id+missile_id) % 1000)

        if n_bombs == 1:
            n_p, n_i = 25, 30
            lb = np.array([0.0, 70.0, 0.0, 0.0])
            ub = np.array([2*np.pi, 140.0, 15.0, 6.0])
            pos = rng.uniform(lb, ub, (n_p, 4))
            pos[0] = list(init_params)
            for i in range(1, min(8, n_p)):
                pos[i] = np.clip(np.array(init_params) + rng.normal(0, [0.3, 15, 2, 1]), lb, ub)
            vel = np.zeros((n_p, 4))
            pbest = pos.copy()

            def fit(x, uid=uav_id, mid=missile_id):
                th, vv, tr, df = x
                if UAVS[uid][2] - 0.5*G*df**2 < 0: return 0.0
                T, _, _, _ = calc_time_fast(mid, uid, th, vv, tr, df, dt_step=0.02, n_samples=100)
                return T

            pf = np.array([fit(p) for p in pos])
            gi = np.argmax(pf)
            gb, gf = pos[gi].copy(), pf[gi]
            for it in range(n_i):
                r1, r2 = rng.random((n_p, 4)), rng.random((n_p, 4))
                vel = 0.6*vel + 1.8*r1*(pbest-pos) + 1.8*r2*(gb-pos)
                pos = np.clip(pos + vel, lb, ub)
                for i in range(n_p):
                    f = fit(pos[i])
                    if f > pf[i]: pf[i], pbest[i] = f, pos[i].copy()
                    if f > gf: gf, gb = f, pos[i].copy()

            th, vv, tr, df = gb
            T, intervals, bp, tb = calc_time_precise(missile_id, uav_id, th, vv, tr, df)
            all_results[uav_id] = (missile_id, [(th, vv, [(tr, df)], T, intervals)])
            total_T += T
            print(f"    最终: T={T:.4f}s")

        else:
            n_p, n_i = 25, 30
            lb = np.array([0.0, 70.0, 0.0, 0.0, 1.0, 0.0])
            ub = np.array([2*np.pi, 140.0, 15.0, 6.0, 30.0, 6.0])
            dim = 6
            pos = rng.uniform(lb, ub, (n_p, dim))
            pos[0] = [init_params[0], init_params[1], init_params[2], init_params[3],
                      init_params[2]+3, 2.0]
            for i in range(1, min(8, n_p)):
                pos[i] = np.clip(pos[0] + rng.normal(0, [0.3, 15, 2, 1, 3, 1]), lb, ub)
            vel = np.zeros((n_p, dim))
            pbest = pos.copy()

            def fit(x, uid=uav_id, mid=missile_id):
                th, vv, tr1, df1, tr2, df2 = x
                if tr2 < tr1 + 1: return 0.0
                if UAVS[uid][2]-0.5*G*df1**2 < 0 or UAVS[uid][2]-0.5*G*df2**2 < 0: return 0.0
                bombs = [(tr1, df1), (tr2, df2)]
                T, _, _ = calc_multi_bomb_time(mid, uid, th, vv, bombs, dt_step=0.02, n_samples=100)
                return T

            pf = np.array([fit(p) for p in pos])
            gi = np.argmax(pf)
            gb, gf = pos[gi].copy(), pf[gi]
            for it in range(n_i):
                r1, r2 = rng.random((n_p, dim)), rng.random((n_p, dim))
                vel = 0.6*vel + 1.8*r1*(pbest-pos) + 1.8*r2*(gb-pos)
                pos = np.clip(pos + vel, lb, ub)
                for i in range(n_p):
                    f = fit(pos[i])
                    if f > pf[i]: pf[i], pbest[i] = f, pos[i].copy()
                    if f > gf: gf, gb = f, pos[i].copy()

            th, vv, tr1, df1, tr2, df2 = gb
            bombs = [(tr1, df1), (tr2, df2)]
            T, singles, merged = calc_multi_bomb_time(missile_id, uav_id, th, vv, bombs,
                                                       dt_step=0.01, n_samples=180)
            all_results[uav_id] = (missile_id, [(th, vv, bombs, T, merged)])
            total_T += T
            print(f"    最终: T={T:.4f}s (弹1={singles[0]:.3f}, 弹2={singles[1]:.3f})")

    print(f"\nQ5结果: 总遮蔽时长={total_T:.4f}s")
    return all_results, total_T


if __name__ == '__main__':
    from optimizer import solve_q2

    # Q2
    print("="*60)
    print("Q2: 单弹优化")
    print("="*60)
    q2_params, q2_T, _, _ = solve_q2(n_particles=25, n_iter=30)

    # Q3
    print("\n"+"="*60)
    print("Q3: 三弹接力(v2)")
    print("="*60)
    q3_theta, q3_v, q3_bombs, q3_T, q3_merged = solve_q3_v2(q2_params)

    # Q4
    print("\n"+"="*60)
    print("Q4: 三机协同(v2)")
    print("="*60)
    q4_results, q4_T, q4_merged = solve_q4_v2(q2_params)

    # Q5
    print("\n"+"="*60)
    print("Q5: 多对多(v2)")
    print("="*60)
    q5_results, q5_T = solve_q5_v2(q4_results)

    # 汇总
    print("\n"+"="*60)
    print("汇总")
    print("="*60)
    print(f"Q1: 1.3920s")
    print(f"Q2: {q2_T:.4f}s")
    print(f"Q3: {q3_T:.4f}s")
    print(f"Q4: {q4_T:.4f}s")
    print(f"Q5: {q5_T:.4f}s")
