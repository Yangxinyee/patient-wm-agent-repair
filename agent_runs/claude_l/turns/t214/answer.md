The realised weights reproduce the intended z token to 1e-11, so the gate and the pass-through work exactly as designed. The copy of u is accurate to 0.1%. Residual sds inside the model are 0.64 (insulin) and 0.24 (NE), matching the earlier decode check.

Next is the factual cost. I'll run the exact teacher-forced replica on the 100 tuning stays of batch 1 with κ = 4 for both drugs. I'll compare each reading's NLL with the current model (`cur`), overall and per code: 5 lab, 6 finger, 7 arterial, 10 cuff. I'll also print the return part of the rollout code, so the next run can compute dose-edit readouts.

**Expectation:** if the gated residual injection is factually cheap, the overall change should be within about ±0.002. The NE part may show a cost on codes 7 and 10, because its residual is mostly policy signal. A cost above about 0.005 overall would mean κ_N must come down or the NE residual needs a better reference.

```python
t0=time.time()
def tf_all(Wx, ks):
    out=[]
    for s in ks:
        for r in tf_stay_z(Wx,BUN[s],None): out.append((s,)+tuple(r))
    return out
res_k4=tf_all(Wk,keys)
curd={(r[0],round(r[1],4),r[2]):r for r in cur}
d=[];cc=[];sh=[]
for r in res_k4:
    q=curd.get((r[0],round(r[1],4),r[2]))
    if q is None: continue
    d.append(r[4]-q[4]); cc.append(r[2]); sh.append(r[3]-q[3])
d=np.array(d);cc=np.array(cc);sh=np.array(sh)
print('matched',len(d),'of',len(res_k4),'dNLL %.5f'%d.mean(),{int(c):(round(float(d[cc==c].mean()),4),int((cc==c).sum())) for c in np.unique(cc)})
print('shift sd by code',{int(c):round(float(sh[cc==c].std()),3) for c in np.unique(cc)},' time %.0fs'%(time.time()-t0))
src=open('/work/wm2_rollout.py').read(); i=src.rfind('return'); print(src[i-700:i+200])
```