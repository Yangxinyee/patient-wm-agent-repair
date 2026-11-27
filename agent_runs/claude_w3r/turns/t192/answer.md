Run 179 gave the step's call sequence, with the wrappers leaving the output unchanged. Every LayerNorm output is small: tokens of norm ~700 per slot come out at ~13–37. So a dose signal injected into the big residual stream is shrunk about 50× by each norm. It is only readable downstream if it lies along a direction where the stream naturally varies very little.

My hypothesis is that the prelude's pre-norm slot tokens (x2) and the state y have such low-variance directions. Then a modest injection that rises with the rate (small relative to σ≈60) would be cleanly readable, with little disturbance to the norms.

To test this I'll collect, for every bundle of the 100 selS stays, the prelude input tokens, the attention pattern, x2, E, mean y and out, plus each bundle's drug start and END rates. The next step is a variance spectrum of these. If the smallest-variance directions have sd ≪ 1, the plan stands. If they're all ≳ 5, I need another route.

```python
import time
def _rates(atoms):
    d={'ins_s':0.,'ins_e':0.,'ne_s':0.,'ne_e':0.,'has_ins_s':0,'has_ne_s':0,'has_ins_e':0,'has_ne_e':0}
    for a in atoms:
        if a['rate'] is None: continue
        nm='ins' if a['code']==8 else 'ne'; kk='_s' if a['kind']==0 else '_e'
        d[nm+kk]+=a['rate'][1]; d['has_'+nm+kk]=1
    return d
CAP={}
def LNt_c(x,name):
    y=_LNt0(x,name)
    if name=='core.prelude.0.norm1': CAP['tok0']=x.clone(); CAP['n1']=y.clone()
    elif name=='core.prelude_norm': CAP['x2']=x.clone(); CAP['E']=y.clone()
    elif name=='core.out_norm': CAP['ym']=x.clone(); CAP['out']=y.clone()
    return y
def mha_c(xq,xk,xv,pre,kadd=None):
    y=_mha0(xq,xk,xv,pre,kadd)
    if pre=='core.prelude.0.attn': CAP['patt']=y[1].clone(); CAP['pout']=y[0].clone()
    return y
t0=time.time(); recsP=[]; ARR={k:[] for k in ['tok0','n1','x2','E','ym','out','patt','pout','hn']}
stays_S=sorted(set(int(s) for s in SST))
globals().update(LNt=LNt_c,mha=mha_c)
try:
    for s in stays_S:
        Bs_,rws,EUs_=stay_data(s)
        for k in range(1,len(Bs_)):
            CAP.clear()
            try: hn,o=step_at(Bs_,rws,EUs_,k)
            except Exception as ex: continue
            for kk in ARR:
                ARR[kk].append((hn if kk=='hn' else CAP[kk]).numpy().astype(np.float32))
            d=_rates(Bs_[k][1]); d.update(stay=s,k=k,t=Bs_[k][0],n_at=len(Bs_[k][1]),dt=Bs_[k][0]-Bs_[k-1][0]); recsP.append(d)
finally:
    globals().update(LNt=_LNt0,mha=_mha0)
P_={k:np.stack(v) for k,v in ARR.items()}; PD=pd.DataFrame(recsP)
print(time.time()-t0, {k:v.shape for k,v in P_.items()}, len(PD))
print(PD[['has_ins_s','has_ne_s','has_ins_e','has_ne_e']].mean())
```