The factual check shows less safety margin than I estimated. On the 100 sel1 tuning stays, the edited model's NLL per reading is 1.59144 against the original's 1.58786, a rise of 0.00358. That is under the 0.005 limit, but only by 0.0014, and my offline estimate was about +0.0015.

The rise comes from three reading types:

| Reading | NLL change per reading | Share of the total rise |
|---|---|---|
| map_arterial | +0.0093 | about +0.0026 |
| glucose_finger and glucose_lab | +0.0038 and +0.0052 | about +0.0018 (insulin repair) |
| map_cuff | −0.0032 | offsets part of it |

Most of the map_arterial cost must come from the norepinephrine ×½ direction. That part was predicted to cost about 0.0028·θₕ².

Before changing anything I need to know which explanation holds:
1. **Different stays:** tuning stays may simply cost more than the train sets I estimated on.
2. **Tool versus replica:** the tool may compute the edited model differently from my offline replica.

To find out, this run computes the offline factual cost on all 400 tuning stays, split into sel1 and the other 300. I do this for the insulin-only model and for three norepinephrine settings, with θₕ = 0.7, 0.35 and 0. In each setting θ₂ is set to keep norepinephrine ×2 at +4.47. I also bootstrap the tool's sel1 cost over stays to see how noisy the estimate is.

I then save the setting with the largest θₕ whose predicted tool cost on all 400 stays is at most 0.0025. The prediction scales the offline cost by the tool-to-offline ratio on sel1, so the margin to the limit stays at least five times its likely error.

What would confirm each explanation:
- **Explanation 1:** offline sel1 also shows about +0.0036.
- **Explanation 2:** offline sel1 shows only about +0.0015.

```python
import time, numpy as np, torch, pandas as pd, traceback as tb
T0=time.time()
fo=pd.read_csv('/work/fwd_6/factual_original.csv'); fe=pd.read_csv('/work/fwd_6/factual_edited.csv')
print(list(fo.columns), len(fo))
tool_c=None
try:
    nc=[c for c in fo.columns if 'nll' in c.lower()][0]
    d=fe[nc].values-fo[nc].values; st_=fo['stay_id'].values
    ps=pd.Series(d).groupby(st_).sum(); cnt=pd.Series(np.ones(len(d))).groupby(st_).sum()
    rng=np.random.default_rng(0); tot=[]
    for b in range(2000):
        s=rng.choice(len(ps),len(ps)); tot.append(ps.values[s].sum()/cnt.values[s].sum())
    tool_c=float(d.mean()); print('tool sel1 cost',round(tool_c,5),'boot se',round(float(np.std(tot)),5),'top stays',ps.sort_values().tail(4).round(2).to_dict())
except Exception: print('csv err',tb.format_exc()[-400:])
try:
    dsT=build_ds(tu,ids,0); ds0=reenc(dsT,PO); ds1=reenc(dsT,Pcur)
    ev=ds0['rd'].event.values; stT=ds0['rd'].stay_id.values; in1=np.isin(stT,sel['sel1'])
    n0=torch.as_tensor(tf_eval(PO,ds0,False)[0]).double().numpy()
    def rep(nm,P):
        n=torch.as_tensor(tf_eval(P,ds1,False)[0]).double().numpy(); d=n-n0
        by={e[:9]:round(float(d[(ev==e)&in1].mean()),4) for e in ['glucose_finger','glucose_lab','map_arterial','map_cuff']}
        r=(round(float(d.mean()),5),round(float(d[in1].mean()),5),round(float(d[~in1].mean()),5))
        print(nm,'all400/sel1/rest300',r,'sel1 by event',by,round(time.time()-T0)); return r
    r_ins=rep('ins_only',Pcur); cand=[]
    for th2,thh in [(0.508,0.7),(1.07,0.35),(1.62,0.0)]:
        P_,_,_=build_ne2(Pcur,th2,thh); cand.append((th2,thh,rep(f'ne {th2},{thh}',P_),P_))
    ratio=(tool_c/cand[0][2][1]) if tool_c else 1.0; print('tool/offline ratio on sel1',round(ratio,3))
    pick=None
    for th2,thh,r,P_ in cand:
        if r[0]*ratio<=0.0025: pick=(th2,thh,r,P_); break
    if pick is None: pick=cand[-1]; print('WARNING none under 0.0025; taking th_h=0')
    th2,thh,r,P_=pick
    for n in ['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias']: save_edit(n,P_[n])
    print('SAVED',th2,thh,'offline cost all400',r[0],'pred tool',round(r[0]*ratio,5),'edited',len(edited()))
    with open('/work/notes.md','a') as f: f.write(f'- Run291: TOOL factual sel1 (th2 .508 thh .7): +{tool_c:.5f} (MAP_art +.0093, gluc_f +.0038, lab +.0052, cuff -.0032) -> too close to .005. Offline tuning costs (all400/sel1/rest300): ins_only {r_ins}; '+'; '.join(f'({a},{b}) {c}' for a,b,c,_ in cand)+f'. tool/offline ratio {ratio:.2f}. SAVED th2={th2} thh={thh} (pred tool cost {r[0]*ratio:.5f}).\n')
except Exception: print('err',tb.format_exc()[-900:])
print('sec',round(time.time()-T0))
```