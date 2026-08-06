"""
Q5 优化版:
1. FY4尝试所有3枚导弹,选最优
2. 每架UAV搜索第2弹(接力)
3. 并集计算总遮蔽时间
"""
import numpy as np, json, sys
from core import *
from optimizer import calc_time_fast, calc_time_precise, calc_multi_bomb_time

def log(msg): print(msg, flush=True)

def search_uav_missile(uav_id, missile_id, v=140):
    """3维网格搜索: theta x t_rel x dt_fuse"""
    uav0 = UAVS[uav_id]
    best_T, best_p = 0.0, None
    for df in np.arange(0, 9.5, 1.0):
        if uav0[2] - 4.9*df**2 < 0: break
        for th_deg in range(0, 360, 4):
            th = np.radians(th_deg)
            for tr in np.arange(0, 45, 1.0):
                T, _, _, _ = calc_time_fast(missile_id, uav_id, th, v, tr, df,
                                             dt_step=0.05, n_samples=50)
                if T > best_T:
                    best_T = T
                    best_p = (th, v, tr, df)
    return best_p, best_T

def pso_refine(uav_id, missile_id, init_p, n_p=25, n_i=30):
    th0, v0, tr0, df0 = init_p
    rng = np.random.RandomState(hash(uav_id+missile_id)%1000)
    lb = np.array([th0-0.3, 70, max(0,tr0-3), max(0,df0-1.5)])
    ub = np.array([th0+0.3, 140, tr0+3, df0+1.5])
    pos = rng.uniform(lb, ub, (n_p, 4))
    pos[0] = list(init_p)
    vel = np.zeros((n_p, 4))
    pbest = pos.copy()
    def fit(x):
        th, vv, tr, df = x
        if UAVS[uav_id][2]-0.5*G*df**2 < 0: return 0.0
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

def search_2nd_bomb(uav_id, missile_id, theta, v, bomb1_params, t_rel_min):
    """搜索第2弹: 固定theta和v, 搜索t_rel2和dt_fuse2"""
    best_T_union = 0.0
    best_params = None
    tr1, df1 = bomb1_params

    for df2 in np.arange(0, 9.5, 1.0):
        if UAVS[uav_id][2] - 4.9*df2**2 < 0: break
        for tr2 in np.arange(t_rel_min, 50, 1.0):
            if tr2 < tr1 + 1.0: continue
            bombs = [(tr1, df1), (tr2, df2)]
            T_union, _, _ = calc_multi_bomb_time(missile_id, uav_id, theta, v, bombs,
                                                  dt_step=0.05, n_samples=50)
            if T_union > best_T_union:
                best_T_union = T_union
                best_params = (tr2, df2)

    if best_params:
        # PSO精修第2弹
        tr2_0, df2_0 = best_params
        rng = np.random.RandomState(hash(uav_id+"2nd")%1000)
        lb2 = np.array([max(t_rel_min, tr2_0-2), max(0, df2_0-1)])
        ub2 = np.array([tr2_0+2, df2_0+1])
        pos2 = rng.uniform(lb2, ub2, (20, 2))
        pos2[0] = [tr2_0, df2_0]
        vel2 = np.zeros((20, 2))
        pb2 = pos2.copy()
        def fit2(x):
            tr2, df2 = x
            if tr2 < tr1 + 1: return 0.0
            if UAVS[uav_id][2]-0.5*G*df2**2 < 0: return 0.0
            bombs = [(tr1, df1), (tr2, df2)]
            T, _, _ = calc_multi_bomb_time(missile_id, uav_id, theta, v, bombs, dt_step=0.02, n_samples=80)
            return T
        pf2 = np.array([fit2(p) for p in pos2])
        gi2 = np.argmax(pf2); gb2, gf2 = pos2[gi2].copy(), pf2[gi2]
        for it in range(25):
            r1, r2 = rng.random((20,2)), rng.random((20,2))
            vel2 = 0.6*vel2 + 1.8*r1*(pb2-pos2) + 1.8*r2*(gb2-pos2)
            pos2 = np.clip(pos2+vel2, lb2, ub2)
            for i in range(20):
                f = fit2(pos2[i])
                if f > pf2[i]: pf2[i], pb2[i] = f, pos2[i].copy()
                if f > gf2: gf2, gb2 = f, pos2[i].copy()
        best_params = tuple(gb2)
        bombs = [(tr1, df1), best_params]
        best_T_union, _, _ = calc_multi_bomb_time(missile_id, uav_id, theta, v, bombs, dt_step=0.01, n_samples=180)

    return best_params, best_T_union


if __name__ == '__main__':
    with open(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\results.json") as f:
        R = json.load(f)

    log("="*60)
    log("Q5 优化: 任务重分配 + 第2弹接力")
    log("="*60)

    # Step1: FY4尝试所有导弹
    log("\n--- FY4 任务重分配 ---")
    fy4_best_mid, fy4_best_T, fy4_best_p = None, 0, None
    for mid in ['M1', 'M2', 'M3']:
        log(f"  FY4→{mid} 搜索...")
        p, T = search_uav_missile('FY4', mid)
        if p and T > 0.01:
            p2, T2 = pso_refine('FY4', mid, p)
            Tp, _, _, _ = calc_time_precise(mid, 'FY4', *p2)
            log(f"    T={Tp:.4f}s, θ={np.degrees(p2[0]):.2f}°")
            if Tp > fy4_best_T:
                fy4_best_T, fy4_best_mid, fy4_best_p = Tp, mid, p2
        else:
            log(f"    无有效解")

    if fy4_best_mid:
        log(f"  FY4 最佳分配: →{fy4_best_mid}, T={fy4_best_T:.4f}s")
    else:
        log(f"  FY4 无法遮蔽任何导弹")

    # Step2: 组装第1弹分配
    assign = {
        'FY1': ('M1', R['Q5_params']['FY1']['params']),
        'FY2': ('M1', R['Q5_params']['FY2']['params']),
        'FY3': ('M2', R['Q5_params']['FY3']['params']),
        'FY5': ('M3', R['Q5_params']['FY5']['params']),
    }
    if fy4_best_mid:
        assign['FY4'] = (fy4_best_mid, list(fy4_best_p))

    # Step3: 每架UAV搜索第2弹
    log("\n--- 第2弹接力搜索 ---")
    final_results = {}
    for uid, (mid, p1) in assign.items():
        theta, v, tr1, df1 = p1
        log(f"  {uid}→{mid} 第2弹搜索...")
        T1, iv1, _, _ = calc_time_precise(mid, uid, theta, v, tr1, df1)
        log(f"    第1弹: T={T1:.4f}s")

        t_rel_min = tr1 + 1.0
        bomb2, T_union = search_2nd_bomb(uid, mid, theta, v, (tr1, df1), t_rel_min)

        if bomb2 and T_union > T1 + 0.1:
            log(f"    第2弹: tr={bomb2[0]:.3f}, df={bomb2[1]:.3f}")
            log(f"    并集: T={T_union:.4f}s (增量={T_union-T1:.4f}s)")
            final_results[uid] = (mid, theta, v, [(tr1, df1), bomb2], T_union)
        else:
            log(f"    第2弹无增量")
            final_results[uid] = (mid, theta, v, [(tr1, df1)], T1)

    # Step4: 计算总遮蔽时间
    log("\n--- 最终结果 ---")
    total_T = 0.0
    for uid in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
        if uid in final_results:
            mid, th, v, bombs, T = final_results[uid]
            total_T += T
            log(f"  {uid}→{mid}: {len(bombs)}弹, T={T:.4f}s")
        else:
            log(f"  {uid}: 未分配")

    log(f"\n{'='*60}")
    log(f"Q5优化总遮蔽: {total_T:.4f}s (之前: 15.149s)")
    log(f"{'='*60}")

    # 保存优化结果
    opt_results = {
        'Q5_optimized': round(total_T, 4),
        'details': {uid: {
            'missile': final_results[uid][0],
            'theta_deg': round(np.degrees(final_results[uid][1])%360, 4),
            'v': round(final_results[uid][2], 4),
            'bombs': [[round(b[0],4), round(b[1],4)] for b in final_results[uid][3]],
            'T': round(final_results[uid][4], 4)
        } for uid in final_results}
    }
    with open(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\q5_optimized.json", 'w') as f:
        json.dump(opt_results, f, indent=2, ensure_ascii=False)
    log("q5_optimized.json 已保存")
