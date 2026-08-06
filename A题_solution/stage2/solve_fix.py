"""
FY2~FY5 修复: 扩大dt_fuse范围到10s + 逆向运动学初始解
"""
import numpy as np, sys
from core import *
from optimizer import calc_time_fast, calc_time_precise

def log(msg): print(msg, flush=True)

def search_uav(uav_id, missile_id):
    """3维网格: theta x t_rel x dt_fuse, v=140"""
    uav0 = UAVS[uav_id]
    best_T, best_p = 0.0, None

    # 搜索dt_fuse从0到9, 步长1s
    for df in np.arange(0, 9.5, 1.0):
        burst_z = uav0[2] - 4.9 * df ** 2
        if burst_z < 0:
            break
        for th_deg in range(0, 360, 4):
            th = np.radians(th_deg)
            for tr in np.arange(0, 40, 1.0):
                T, _, _, _ = calc_time_fast(missile_id, uav_id, th, 140.0, tr, df,
                                             dt_step=0.05, n_samples=50)
                if T > best_T:
                    best_T = T
                    best_p = (th, 140.0, tr, df)
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
        if tr < 0 or df < 0: return 0.0
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
    from optimizer import solve_q2
    from solve_v2 import solve_q3_v2

    # Q2
    log("="*50); log("Q2"); log("="*50)
    q2p, q2T, _, _ = solve_q2(n_particles=25, n_iter=30)
    log(f"Q2: {q2T:.4f}s")

    # Q3
    log("\nQ3...")
    q3th, q3v, q3bombs, q3T, _ = solve_q3_v2(q2p)
    log(f"Q3: {q3T:.4f}s")

    # Q4: FY2, FY3 扩大dt_fuse搜索
    log("\n"+"="*50); log("Q4 (扩大dt_fuse)"); log("="*50)
    q4 = {'FY1': tuple(q2p)}
    for uid in ['FY2', 'FY3']:
        log(f"  {uid} 3维网格搜索(dt_fuse 0~9)...")
        p, T = search_uav(uid, 'M1')
        if p and T > 0.01:
            log(f"    网格: T={T:.3f}s, θ={np.degrees(p[0]):.1f}°, v={p[1]:.0f}, tr={p[2]:.1f}, df={p[3]:.1f}")
            log(f"    PSO精修...")
            p2, T2 = pso_refine(uid, 'M1', p)
            Tp, iv, _, _ = calc_time_precise('M1', uid, *p2)
            q4[uid] = p2
            log(f"    {uid}: T={Tp:.4f}s, θ={np.degrees(p2[0]):.2f}°, v={p2[1]:.1f}, tr={p2[2]:.3f}, df={p2[3]:.3f}")
        else:
            q4[uid] = (0.0, 100.0, 1.0, 0.0)
            log(f"    {uid}: 无有效遮蔽")

    # Q4并集
    all_iv = []
    for uid, (th,vv,tr,df) in q4.items():
        T, iv, _, _ = calc_time_precise('M1', uid, th, vv, tr, df)
        all_iv.extend(iv)
        log(f"  {uid}: T={T:.4f}s")
    all_iv.sort()
    merged = [list(all_iv[0])] if all_iv else []
    for s,e in all_iv[1:]:
        if s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([s,e])
    q4T = sum(e-s for s,e in merged) if merged else 0
    log(f"Q4联合: {q4T:.4f}s")

    # Q5
    log("\n"+"="*50); log("Q5"); log("="*50)
    assign = {'FY1':'M1', 'FY2':'M1', 'FY3':'M2', 'FY4':'M3', 'FY5':'M3'}
    q5 = {}; q5T = 0.0
    for uid, mid in assign.items():
        log(f"  {uid}→{mid} 搜索...")
        if uid == 'FY1':
            # FY1用Q2结果
            T, iv, _, _ = calc_time_precise(mid, uid, *q2p)
            q5[uid] = (mid, tuple(q2p), T, iv)
            q5T += T
            log(f"    T={T:.4f}s")
        else:
            p, T = search_uav(uid, mid)
            if p and T > 0.01:
                p2, T2 = pso_refine(uid, mid, p)
                Tp, iv, _, _ = calc_time_precise(mid, uid, *p2)
                q5[uid] = (mid, p2, Tp, iv)
                q5T += Tp
                log(f"    T={Tp:.4f}s, θ={np.degrees(p2[0]):.2f}°")
            else:
                q5[uid] = (mid, (0,100,1,0), 0, [])
                log(f"    无有效遮蔽")
    log(f"Q5总计: {q5T:.4f}s")

    log("\n"+"="*50)
    log("汇总")
    log("="*50)
    log(f"Q1: 1.3920s")
    log(f"Q2: {q2T:.4f}s")
    log(f"Q3: {q3T:.4f}s")
    log(f"Q4: {q4T:.4f}s")
    log(f"Q5: {q5T:.4f}s")

    # 保存结果供论文使用
    import json
    results = {
        'Q1': 1.3920, 'Q2': round(q2T,4), 'Q3': round(q3T,4),
        'Q4': round(q4T,4), 'Q5': round(q5T,4),
        'Q2_params': [round(float(x),4) for x in q2p],
        'Q3_params': [round(np.degrees(q3th),4), round(q3v,4),
                      [[round(tr,4),round(df,4)] for tr,df in q3bombs]],
        'Q4_params': {uid: [round(float(x),4) for x in q4[uid]] for uid in q4},
        'Q5_params': {uid: {'missile': q5[uid][0],
                            'params': [round(float(x),4) for x in q5[uid][1]],
                            'T': round(q5[uid][2],4)} for uid in q5},
    }
    with open(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("results.json 已保存")
