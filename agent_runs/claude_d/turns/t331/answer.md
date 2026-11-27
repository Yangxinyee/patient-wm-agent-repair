E7 on sel1 costs +0.0020 overall (1.51393 against 1.51191), as predicted.
- glucose_finger +0.0065, glucose_lab −0.0003, map_arterial +0.0007, map_cuff −0.0015.
- Compared with E6 on the same stays, the 0.4 NE channel adds about +0.0046 per map_arterial reading.

Pooled over all 400 tuning stays, E7 comes to about −0.0002, essentially zero. The four 100-stay sets range from −0.0018 to +0.0020, a spread of about 0.0018. sel1 is the worst set.

That leaves room to raise the gains. My sel1 and sel2 readout checks suggest:
- **Insulin at full gain (f = 1)** matches my effect estimates: −13.9/+7.1 against −14.0/+7.0, and −13.4/+6.1 against −13.2/+6.6.
- **NE** needs about 0.9 to get x1/2 right, and more for x2.

I will project the factual cost of every pair of gains from the per-reading data of all 400 stays:
- **Glucose readings:** cost = c2·Σshift²·f² − c1·Σshift·residual·f. I calibrate this on sel1 at f = 0.8 and f = 1 and check it against R1–R3.
- **Map readings:** a quadratic in f_ne plus a fixed part, calibrated on sel1 at f_ne = 0, 0.4 and 1.
- **Readouts:** taken as linear in each gain, from the sel1 and sel2 dose_edit runs.

I will then pick the pair that minimises the readout error subject to two limits: projected pooled cost ≤ 0.0020 and worst set ≤ 0.0042. I save that pair as E8.

Verification plan:
- **Cost model:** it is confirmed if its predicted glucose costs for R1–R3 match the measured ones (0.0011, 0.0013, 0.0005).
- **E8 itself:** factual checks on the worst set and pooled, then a dose_edit run.

```python
import numpy as np, pandas as pd, json, torch
def ld(k):
    a=pd.read_csv(f'/work/fwd_{k}/factual_original.csv'); b=pd.read_csv(f'/work/fwd_{k}/factual_edited.csv')
    return pd.DataFrame({'ev':a.event,'dn':b.nll-a.nll,'dp':b.predicted_mean-a.predicted_mean,'e0':a.observed-a.predicted_mean})
def parts(d,fi=.8,fn=.4):
    g=d.ev.str.startswith('glucose').values; N=len(d)
    dpg=d.dp.values[g]/fi; dpm=d.dp.values[~g]/fn
    return dict(N=N,G2=(dpg**2).sum()/N,G1=(dpg*d.e0.values[g]).sum()/N,M2=(dpm**2).sum()/N,M1=(dpm*d.e0.values[~g]).sum()/N,cg=d.dn.values[g].sum()/N,cm=d.dn.values[~g].sum()/N)
P={s:parts(ld(k)) for s,k in {'sel1':21,'R1':18,'R2':19,'R3':20}.items()}
d16=ld(16); g16=d16.ev.str.startswith('glucose').values; d17=ld(17); g17=d17.ev.str.startswith('glucose').values
cg1=d16.dn.values[g16].sum()/len(d16); cm1=d16.dn.values[~g16].sum()/len(d16); cm0=d17.dn.values[~g17].sum()/len(d17)
p=P['sel1']; c2,c1=np.linalg.solve([[p['G2']*.64,-p['G1']*.8],[p['G2'],-p['G1']]],[p['cg'],cg1])
m2,m1,gam=np.linalg.solve([[0,0,1],[p['M2']*.16,-p['M1']*.4,1],[p['M2'],-p['M1'],1]],[cm0,p['cm'],cm1])
print('c2 %.4g c1 %.4g m2 %.4g m1 %.4g gam1 %.5f'%(c2,c1,m2,m1,gam))
for s,q in P.items():
    q['gam']=q['cm']-(m2*q['M2']*.16-m1*q['M1']*.4)
    print(s,'gluc pred %.5f meas %.5f | map meas %.5f gam %.5f | M2 %.3f'%(c2*q['G2']*.64-c1*q['G1']*.8,q['cg'],q['cm'],q['gam'],q['M2']))
cost=lambda q,fi,fn: c2*q['G2']*fi*fi-c1*q['G1']*fi+m2*q['M2']*fn*fn-m1*q['M1']*fn+q['gam']
Nt=sum(q['N'] for q in P.values())
s14=json.load(open('/work/fwd_14/summary.json'))['results']; s13=json.load(open('/work/fwd_13/summary.json'))['results']
K=['insulin_x2','insulin_half','ne_x2','ne_half']
O={'s1':[s13['original']['edits'][k]['mean_shift'] for k in K],'s2':[s14['original']['edits'][k]['mean_shift'] for k in K]}
E={'s1':[s13['edited']['edits'][k]['mean_shift'] for k in K],'s2':[s14['edited']['edits'][k]['mean_shift'] for k in K]}
T={'s1':[-13.99,6.99,4.85,-2.43],'s2':[-13.16,6.58,3.78,-1.89]}
def ro(fi,fn): return {s:[O[s][j]+(fi if j<2 else fn)*(E[s][j]-O[s][j]) for j in range(4)] for s in O}
def sc(fi,fn):
    r=ro(fi,fn); w=[2,1,1,1]; return sum(w[j]*abs(r[s][j]-T[s][j])/abs(T[s][j]) for s in r for j in range(4))/10
best=None
for fi in (.8,.85,.9,.95,1.0):
    row=[]
    for fn in (.4,.5,.6,.7,.8,.9,1.0):
        cs={s:cost(q,fi,fn) for s,q in P.items()}; pool=sum(cs[s]*P[s]['N'] for s in cs)/Nt
        ok=pool<=.0020 and max(cs.values())<=.0042; row.append('%.1f:%.4f/%.4f%s'%(fn,pool,max(cs.values()),'*' if ok else ''))
        if ok and (best is None or sc(fi,fn)<best[0]): best=(sc(fi,fn),fi,fn,pool,max(cs.values()))
    print('fi',fi,' '.join(row))
print('orig sel2', np.round(O['s2'],2), 'best', np.round(best,4), 'readouts', {s:np.round(v,2) for s,v in ro(best[1],best[2]).items()})
_,fi,fn,_,_=best
z3=np.load('/work/E3b.npz'); z4=np.load('/work/E4.npz'); z7=np.load('/work/E7h.npz')
W1,b1,W2,b2=[z7[f'arr_{i}'].astype(np.float32).copy() for i in range(4)]
bf=lambda x: torch.tensor(x).to(torch.bfloat16).float().numpy()
W2[:,[31,143]]=bf(z3['arr_2'][:,[31,143]]*fi); W2[:,[124,142]]=bf(z4['arr_2'][:,[124,142]]*fn)
hn=['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias']
for n,a in zip(hn,[W1,b1,W2,b2]): save_edit(n, torch.from_numpy(np.ascontiguousarray(a)))
np.savez('/work/E8h.npz',W1,b1,W2,b2)
with open('/work/notes.md','a') as fh: fh.write("- E7 factual: sel1 +.0020, R1 -.0016, R2 +.0004, R3 -.0018 (pooled ~-.0002). Projection model (gluc c2/c1 on sel1 f.8&1; map quad on sel1 fne 0/.4/1) -> E8 = ins f=%.2f, NE f=%.2f, proj pooled %.4f max set %.4f. /work/E8h.npz saved as current.\n"%(fi,fn,best[3],best[4]))
print('saved E8', fi, fn)
```