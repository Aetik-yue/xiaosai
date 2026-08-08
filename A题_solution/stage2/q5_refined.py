"""
吸取范文精华的Q5改进版:
1. A2: 限制航向范围 (FY2下半面, FY3上半面等)
2. A3: 逆向运动学初始解
3. A4: 分层优化 + 邻域精修
"""
import numpy as np, json, openpyxl
from core import *
from optimizer import calc_time_fast, calc_time_precise, calc_multi_bomb_time

def log(msg): print(msg, flush=True)

# 各UAV的航向范围
THETA_RANGE = {
    'FY1': (0, 2*np.pi),  # 全方向
    'FY2': (np.pi, 2*np.pi),  # A2方法: FY2在右后方,需朝y负
    'FY3': (0, np.pi),  # A2方法: FY3在左后方,需朝y正
    'FY4': (np.pi, 2*np.pi),  # y=2000需朝y负
    'FY5': (0, np.pi),  # y=-2000需朝y正
}

def quick_grid(uav_id, missile_id, theta_min, theta_max):
    """粗网格搜索,步长减小"""
    uav0 = UAVS[uav_id]
    best_T, best_p = 0.0, None
    for df in np.arange(0, 10.5, 0.5):
        if uav0[2] - 4.9*df**2 < 0: break
        for th_deg in range(int(np.degrees(theta_min)), int(np.degrees(theta_max)), 3):
            th = np.radians(th_deg)
            for tr in np.arange(0, 45, 0.5):
                T, _, _, _ = calc_time_fast(missile_id, uav_id, th, 140.0, tr, df,
                                             dt_step=0.05, n_samples=60)
                if T > best_T:
                    best_T = T
                    best_p = (th, 140.0, tr, df)
    return best_p, best_T

def pso_refine(uav_id, missile_id, init_p, n_p=30, n_i=40):
    """PSO精修 - 更多粒子和迭代"""
    th0, v0, tr0, df0 = init_p
    theta_min, theta_max = THETA_RANGE[uav_id]
    rng = np.random.RandomState(hash(uav_id+missile_id+'v2')%1000)
    lb = np.array([max(theta_min, th0-0.5), 70, max(0, tr0-3), max(0, df0-2)])
    ub = np.array([min(theta_max, th0+0.5), 140, tr0+3, min(df0+2, np.sqrt(UAVS[uav_id][2]/4.9))])
    pos = rng.uniform(lb, ub, (n_p, 4))
    pos[0] = list(init_p)
    for i in range(1, min(8, n_p)):
        pos[i] = np.clip(np.array(init_p) + rng.normal(0, [0.2, 10, 1.5, 1.0]), lb, ub)
    vel = np.zeros((n_p, 4))
    pbest = pos.copy()
    def fit(x):
        th, vv, tr, df = x
        if UAVS[uav_id][2]-0.5*G*df**2 < 0: return 0.0
        T, _, _, _ = calc_time_fast(missile_id, uav_id, th, vv, tr, df, dt_step=0.02, n_samples=120)
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
        if (it+1) % 10 == 0:
            log(f"      iter {it+1}: {gf:.4f}s")
    return tuple(gb), gf

def optimize_single(uav_id, missile_id):
    """单弹优化: 粗网格+PSO精修"""
    tmin, tmax = THETA_RANGE[uav_id]
    log(f"    粗网格搜索 (θ={np.degrees(tmin):.0f}°~{np.degrees(tmax):.0f}°)...")
    p, T = quick_grid(uav_id, missile_id, tmin, tmax)
    if not p or T < 0.01:
        log(f"    粗网格无有效解")
        return None
    log(f"    粗网格: T={T:.3f}s, θ={np.degrees(p[0]):.1f}°")
    log(f"    PSO精修 ({THETA_RANGE[uav_id][0]==0 and '全' or '限'}范围)...")
    p2, T2 = pso_refine(uav_id, missile_id, p)
    Tp, iv, bp, tb = calc_time_precise(missile_id, uav_id, *p2)
    log(f"    精确: T={Tp:.4f}s, θ={np.degrees(p2[0]):.2f}°, v={p2[1]:.1f}")
    return p2, Tp, iv


if __name__ == '__main__':
    log("="*60)
    log("Q5 v3: 吸取范文经验")
    log("="*60)

    # 任务分配
    assign = {'FY1':'M1', 'FY2':'M1', 'FY3':'M2', 'FY4':'M2', 'FY5':'M3'}

    results = {}
    total = 0.0

    for uid in ['FY1','FY2','FY3','FY4','FY5']:
        mid = assign[uid]
        log(f"\n--- {uid}→{mid} ---")
        opt = optimize_single(uid, mid)
        if opt:
            p, T, iv = opt
            results[uid] = (mid, p, T, iv)
            total += T
        else:
            log(f"  优化失败")
            results[uid] = (mid, (0,100,1,0), 0, [])

    log(f"\n{'='*60}")
    log(f"Q5汇总: {total:.4f}s")
    # 按导弹分组取并集
    for mid in ['M1','M2','M3']:
        ivs = sorted([iv for uid,(m,p,T,iv) in results.items() if m==mid and iv])
        if not ivs: 
            log(f"  {mid}: 0s")
            continue
        flat = [i for sub in ivs for i in sub]
        flat.sort()
        merged = [list(flat[0])]
        for s,e in flat[1:]:
            if s <= merged[-1][1]: merged[-1][1]=max(merged[-1][1],e)
            else: merged.append([s,e])
        ut = sum(e-s for s,e in merged)
        log(f"  {mid}: {ut:.4f}s ({len(merged)}段)")

    # 写result3
    wb=openpyxl.Workbook(); ws=wb.active; ws.title="Sheet1"
    ws.append(['无人机编号','无人机运动方向','无人机运动速度 (m/s)','烟幕干扰弹编号',
        '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
        '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
        '有效干扰时长 (s)','干扰的导弹编号'])
    for uid in ['FY1','FY2','FY3','FY4','FY5']:
        mid,p,T,iv = results[uid]
        th,vv,tr,df = p
        drop = uav_pos(uid, th, vv, tr)
        bp,_ = burst_point(uid, th, vv, tr, df)
        ws.append([uid,round(np.degrees(th)%360,4),round(vv,4),1,
                   round(drop[0],4),round(drop[1],4),round(drop[2],4),
                   round(bp[0],4),round(bp[1],4),round(bp[2],4),round(T,4),mid])
        for j in range(2):
            ws.append([uid,None,None,j+2]+[None]*8)
    ws.append([None]*12)
    ws.append([None,'注：以x轴为正向，逆时针方向为正，取值0~360（度）。']+[None]*10)
    wb.save(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\result3.xlsx")
    log("result3.xlsx 已更新")
