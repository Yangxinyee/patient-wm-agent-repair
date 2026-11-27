The offline evaluator agrees with the tool. On sel1 it gives a factual cost of +0.00376, against the tool's +0.00358, so I can trust it for costs.

sel1 is an expensive set of stays, not a typical one:

| Setting | All 400 tuning stays | sel1 (100) | Other 300 |
|---|---|---|---|
| Current edit, θₕ = 0.7 | +0.0012 | +0.0036 | +0.0002 |
| Insulin only | −0.0003 | | |

So the insulin repair alone actually improves factual fit on average. The norepinephrine ×½ direction adds about 0.0015 at θₕ = 0.7, but only about 0.0004 at θₕ = 0.35.

The cost per stay is heavy-tailed. A few stays add about 2 nats each, and the bootstrap standard error on 100 stays is 0.0014. Whether θₕ = 0.7 is safe depends on how large the unseen evaluation set is and how often bad stays land in it.

The run's automatic rule re-saved θₕ = 0.7. Before accepting that, I want a real estimate of the risk of crossing the limit. This run:
1. Pools per-stay cost differences over 1,000 stays (400 tuning and 600 train).
2. Bootstraps evaluation sets of 100, 200 and 400 stays for θₕ = 0, 0.35, 0.5 and 0.7.
3. Saves the largest θₕ whose 99.5th-percentile cost for a 100-stay set is below 0.0045. For that θₕ, θ₂ is set to keep norepinephrine ×2 at +4.47.
4. Checks the saved setting's four readouts with an offline rollout on 400 stays, correcting θ₂ once if ×2 drifts.

What would settle the choice:
- **Keep θₕ = 0.7** if its 99.5th percentile stays below 0.0045.
- **Step down** if the tail risk is clearly higher. I give up some ×½ improvement to protect the factual limit.

```python
import numpy as np, torch, pandas as pd, time, traceback as tb
T0=time.time()
try:
    sets=[('tu',ds0,ds1)]+[(nm,reenc(d,PO),reenc(d,Pcur)) for nm,d in [('E',dsE),('F',dsF)]]
    base={nm:torch.as_tensor(tf_eval(PO,a,False)[0]).double().numpy() for nm,a,b in sets}
    stays={nm:a['rd'].stay_id.values for nm,a,b in sets}
    cfgs=[(1.62,0.0),(1.07,0.35),(0.83,0.5),(0.508,0.7)]; out={}
    rng=np.random.default_rng(1)
    for th2,thh in cfgs:
        P_,_,_=build_ne2(Pcur,th2,thh); T=[];C=[]
        for nm,a,b in sets:
            d=torch.as_tensor(tf_eval(P_,b,False)[0]).double().numpy()-base[nm]
            g=pd.DataFrame({'s':stays[nm],'d':d}).groupby('s')['d'].agg(['sum','count']); T+=g['sum'].tolist(); C+=g['count'].tolist()
        T=np.array(T); C=np.array(C); row={'pop':round(T.sum()/C.sum(),5)}
        for N in [100,200,400]:
            idx=rng.integers(0,len(T),(4000,N)); cs=T[idx].sum(1)/C[idx].sum(1)
            row[N]=(round(float(np.percentile(cs,99)),4),round(float(np.percentile(cs,99.5)),4),round(float((cs>0.005).mean()),4))
        out[(th2,thh)]=(row,P_); print('cfg',th2,thh,'n stays',len(T),row,round(time.time()-T0))
    pick=None
    for th2,thh in reversed(cfgs):
        if out[(th2,thh)][0][100][1]<0.0045: pick=(th2,thh); break
    if pick is None: pick=cfgs[0]
    th2,thh=pick; print('PICK',pick)
    for it in range(2):
        P_,_,_=build_ne2(Pcur,th2,thh); R=eval_shifts(P_,ids,20,0); rr={k:round(float(v[0]),2) for k,v in R.items()}; print('th2',round(th2,3),'thh',thh,rr,round(time.time()-T0))
        if abs(rr['norepinephrine_x2']-4.47)<0.2 or it==1: break
        th2=th2+(4.47-rr['norepinephrine_x2'])/3.13
    for n in ['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias']: save_edit(n,P_[n])
    NEsol=(th2,thh); print('SAVED',round(th2,3),thh,rr,'edited',len(edited()))
    with open('/work/notes.md','a') as f: f.write(f'- Run292: per-stay cost bootstrap over 1000 stays (400 tuning+600 train), (p99,p99.5,P>.005) for N=100/200/400: '+'; '.join(f'{k}: {v[0]}' for k,v in out.items())+f'. PICK th2={th2:.3f} thh={thh} -> offline 400 tuning {rr}. SAVED.\n')
except Exception: print('err',tb.format_exc()[-900:])
print('sec',round(time.time()-T0))
```