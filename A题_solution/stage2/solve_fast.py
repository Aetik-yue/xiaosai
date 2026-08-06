"""
高效版 Q2~Q5 + result输出
- Q4/Q5: 固定v=140, dt_fuse=0, 只搜theta和t_rel(2维粗网格+PSO精修)
- 加flush确保看到进度
"""
import numpy as np, openpyxl, sys
from core import *
from optimizer import calc_time_fast, calc_time_precise, calc_multi_bomb_time, solve_q2

def log(msg):
    print(msg, flush=True)

# ============ Q4: 快速网格搜索 ============

def quick_search(uav_id, missile_id, theta_range, t_rel_range):
    """2维网格: v=140, dt_fuse=0, 粗步长"""
    best_T, best_p = 0.0, None
    for th_deg in theta_range:
        th = np.radians(th_deg)
        for tr in t_rel_range:
            T, _, _, _ = calc_time_fast(missile_id, uav_id, th, 140.0, tr, 0.0,
                                         dt_step=0.05, n_samples=60)
            if T > best_T:
                best_T = T
                best_p = (th, 140.0, tr, 0.0)
    return best_p, best_T

def pso_refine(uav_id, missile_id, init_p, n_p=20, n_i=25):
    """PSO精修"""
    th0, v0, tr0, df0 = init_p
    rng = np.random.RandomState(hash(uav_id+missile_id)%1000)
    lb = np.array([th0-0.5, 70, max(0,tr0-3), 0])
    ub = np.array([th0+0.5, 140, tr0+3, 3])
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


def solve_all_fast():
    # Q2
    log("="*50); log("Q2"); log("="*50)
    q2p, q2T, _, _ = solve_q2(n_particles=25, n_iter=30)
    log(f"Q2: {q2T:.4f}s")

    # Q3
    log("\n"+"="*50); log("Q3"); log("="*50)
    from solve_v2 import solve_q3_v2
    q3th, q3v, q3bombs, q3T, _ = solve_q3_v2(q2p)
    log(f"Q3: {q3T:.4f}s")

    # Q4: FY2, FY3 快速搜索
    log("\n"+"="*50); log("Q4"); log("="*50)
    q4 = {'FY1': tuple(q2p)}

    for uid in ['FY2', 'FY3']:
        log(f"  {uid} 网格搜索...")
        # 全方向搜索,步长5度, t_rel步长1s
        p, T = quick_search(uid, 'M1', range(0, 360, 5), np.arange(0, 50, 1.0))
        log(f"    网格最优: T={T:.3f}s, θ={np.degrees(p[0]):.1f}°" if p else "    无有效解")
        if p and T > 0.01:
            log(f"    PSO精修...")
            p2, T2 = pso_refine(uid, 'M1', p)
            q4[uid] = p2
            Tp, _, _, _ = calc_time_precise('M1', uid, *p2)
            log(f"    {uid}: T={Tp:.4f}s, θ={np.degrees(p2[0]):.2f}°, v={p2[1]:.1f}")
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

    # Q5: 5机各1弹(简化), 分配任务
    log("\n"+"="*50); log("Q5"); log("="*50)
    # 分配: FY1→M1, FY2→M1, FY3→M2, FY4→M3, FY5→M3
    assign = {'FY1':'M1', 'FY2':'M1', 'FY3':'M2', 'FY4':'M3', 'FY5':'M3'}
    q5 = {}
    q5T = 0.0
    for uid, mid in assign.items():
        log(f"  {uid}→{mid} 网格搜索...")
        p, T = quick_search(uid, mid, range(0, 360, 5), np.arange(0, 50, 1.0))
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

    # ============ 写result文件 ============
    log("\n写入result文件...")

    # result1: Q3 (FY1 3弹→M1)
    wb = openpyxl.Workbook(); ws = wb.active; ws.title="Sheet1"
    ws.append(['无人机运动方向','无人机运动速度 (m/s)','烟幕干扰弹编号',
        '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
        '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
        '有效干扰时长 (s)'])
    for j,(tr,df) in enumerate(q3bombs):
        drop = uav_pos('FY1', q3th, q3v, tr)
        bp, _ = burst_point('FY1', q3th, q3v, tr, df)
        Ts, _, _, _ = calc_time_precise('M1','FY1',q3th,q3v,tr,df)
        ws.append([round(np.degrees(q3th)%360,4), round(q3v,4), j+1,
                   round(drop[0],4),round(drop[1],4),round(drop[2],4),
                   round(bp[0],4),round(bp[1],4),round(bp[2],4), round(Ts,4)])
    ws.append([None]*10)
    ws.append(['注：以x轴为正向，逆时针方向为正，取值0~360（度）。']+[None]*9)
    wb.save(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\result1.xlsx")
    log("  result1.xlsx OK")

    # result2: Q4 (3机各1弹→M1)
    wb = openpyxl.Workbook(); ws = wb.active; ws.title="Sheet1"
    ws.append(['无人机编号','无人机运动方向','无人机运动速度 (m/s)',
        '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
        '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
        '有效干扰时长 (s)'])
    for uid in ['FY1','FY2','FY3']:
        th,vv,tr,df = q4[uid]
        drop = uav_pos(uid, th, vv, tr)
        bp, _ = burst_point(uid, th, vv, tr, df)
        T,_,_,_ = calc_time_precise('M1',uid,th,vv,tr,df)
        ws.append([uid, round(np.degrees(th)%360,4), round(vv,4),
                   round(drop[0],4),round(drop[1],4),round(drop[2],4),
                   round(bp[0],4),round(bp[1],4),round(bp[2],4), round(T,4)])
    ws.append([None]*10)
    ws.append([None,'注：以x轴为正向，逆时针方向为正，取值0~360（度）。']+[None]*8)
    wb.save(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\result2.xlsx")
    log("  result2.xlsx OK")

    # result3: Q5 (5机至多3弹→M1M2M3)
    wb = openpyxl.Workbook(); ws = wb.active; ws.title="Sheet1"
    ws.append(['无人机编号','无人机运动方向','无人机运动速度 (m/s)','烟幕干扰弹编号',
        '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
        '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
        '有效干扰时长 (s)','干扰的导弹编号'])
    for uid in ['FY1','FY2','FY3','FY4','FY5']:
        mid, (th,vv,tr,df), T, iv = q5[uid]
        drop = uav_pos(uid, th, vv, tr)
        bp, _ = burst_point(uid, th, vv, tr, df)
        ws.append([uid, round(np.degrees(th)%360,4), round(vv,4), 1,
                   round(drop[0],4),round(drop[1],4),round(drop[2],4),
                   round(bp[0],4),round(bp[1],4),round(bp[2],4), round(T,4), mid])
        for j in range(2):  # 补空行
            ws.append([uid, None, None, j+2]+[None]*8)
    ws.append([None]*12)
    ws.append([None,'注：以x轴为正向，逆时针方向为正，取值0~360（度）。']+[None]*10)
    wb.save(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\result3.xlsx")
    log("  result3.xlsx OK")

    # 汇总
    log("\n"+"="*50)
    log("最终汇总")
    log("="*50)
    log(f"Q1: 1.3920s")
    log(f"Q2: {q2T:.4f}s")
    log(f"Q3: {q3T:.4f}s")
    log(f"Q4: {q4T:.4f}s")
    log(f"Q5: {q5T:.4f}s")

    return q2p, q2T, q3th, q3v, q3bombs, q3T, q4, q4T, q5, q5T


if __name__ == '__main__':
    solve_all_fast()
