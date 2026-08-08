"""
Q5 refin: 在现有最优参数基础上做精细PSO搜索
不对每个UAV从头网格，只在邻域搜索改进
"""
import numpy as np, json, openpyxl
from core import *
from optimizer import calc_time_fast, calc_time_precise

def log(msg): print(msg, flush=True)

def fine_pso(uav_id, missile_id, init_p, n_p=40, n_i=50):
    """细粒度PSO: 在初始解邻域搜索"""
    th0, v0, tr0, df0 = init_p
    rng = np.random.RandomState(hash(uav_id+missile_id+'fine')%1000)
    lb = np.array([th0-0.3, max(70,v0-15), max(0,tr0-2), max(0,df0-1.5)])
    ub = np.array([th0+0.3, min(140,v0+15), tr0+2, min(df0+1.5, np.sqrt(UAVS[uav_id][2]/4.9))])
    pos = rng.uniform(lb, ub, (n_p, 4))
    pos[0] = list(init_p)
    vel = np.zeros((n_p, 4))
    pbest = pos.copy()
    def fit(x):
        th,vv,tr,df = x
        if UAVS[uav_id][2]-0.5*G*df**2 < 0: return 0.0
        T,_,_,_ = calc_time_fast(missile_id,uav_id,th,vv,tr,df,dt_step=0.01,n_samples=150)
        return T
    pf = np.array([fit(p) for p in pos])
    gi = np.argmax(pf); gb,gf=pos[gi].copy(),pf[gi]
    best_iter = 0
    for it in range(n_i):
        r1,r2 = rng.random((n_p,4)),rng.random((n_p,4))
        vel = 0.5*vel + 2.0*r1*(pbest-pos) + 2.0*r2*(gb-pos)
        pos = np.clip(pos+vel, lb, ub)
        for i in range(n_p):
            f = fit(pos[i])
            if f>pf[i]: pf[i],pbest[i]=f,pos[i].copy()
            if f>gf: gf,gb=f,pos[i].copy(); best_iter=it
        if (it+1)%10==0:
            log(f"    iter {it+1}: {gf:.4f}s")
    return tuple(gb), gf

# 加载现有最优
with open(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\q5_updated.json") as f:
    old = json.load(f)

log("="*60)
log("Q5 精细PSO (在现有最优附近搜索)")
log("="*60)

assign = {uid:old['details'][uid]['missile'] for uid in old['details']}
results = {}
total = 0.0

for uid in ['FY1','FY2','FY3','FY4','FY5']:
    mid = assign[uid]
    d = old['details'][uid]
    init_p = (np.radians(d['theta_deg']), d['v'], d['t_rel'], d['dt_fuse'])
    
    log(f"\n--- {uid}→{mid} 精细搜索 ---")
    log(f"  初始: θ={d['theta_deg']:.2f}°, v={d['v']:.1f}, T={d['T']:.3f}s")
    
    p2, T2 = fine_pso(uid, mid, init_p, n_p=35, n_i=45)
    Tp, iv, bp, tb = calc_time_precise(mid, uid, *p2)
    
    delta = Tp - d['T']
    log(f"  精细解: θ={np.degrees(p2[0]):.2f}°, v={p2[1]:.1f}, T={Tp:.4f}s ({'+' if delta>=0 else ''}{delta:.4f}s)")
    
    results[uid] = (mid, p2, Tp, iv)
    total += Tp

log(f"\n{'='*60}")
log(f"Q5精细搜索后: {total:.4f}s (之前: {old['Q5_updated']}s)")
log(f"{'='*60}")

# result3
wb=openpyxl.Workbook(); ws=wb.active; ws.title="Sheet1"
ws.append(['无人机编号','无人机运动方向','无人机运动速度 (m/s)','烟幕干扰弹编号',
    '投放x','投放y','投放z','起爆x','起爆y','起爆z','有效干扰时长 (s)','干扰的导弹编号'])

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
