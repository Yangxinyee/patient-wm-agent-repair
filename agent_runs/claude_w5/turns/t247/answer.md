`fixed_of` is now exact, and `Wt(n)` reads the global `WCUR`, so switching `WCUR` between `W64` and `WE` switches models.

Now the decisive comparison. I'll run the full sequence on the selA tuning stays through 12:05, original and edited, and take the 13:00 glucose_finger predicted mean with mean z. I'll also record the edit's decoded insulin signal, the projection of E on q_I, at drug bundles, to look for an accumulated bias. Before that I'll check that `WE` matches the saved edits.

```python
import time
for nm in edited():
    print(nm, float((WE[nm].float() - load(nm)).abs().max()))
def step_with(Wdict):
    def f(out,h,dt,e,u,Mv,Ma):
        globals()['WCUR'] = Wdict
        r = mystep(dict(out0=out,h0=h,dt=dt,e=e,u=u,Mv=Mv,Ma=Ma))
        return (r['out'], r['y'], r)
    return f
stepO, stepE = step_with(W64), step_with(WE)
TB = {sid: make_bundles(tun[tun.stay_id==sid]) for sid in selA}
def last_before(B, k, code):
    for j in range(k-1, -1, -1):
        for a in B[j][1]:
            if a['code']==code and a['kind']==0 and a.get('value') is not None and not (isinstance(a.get('value'),float) and math.isnan(a['value'])):
                return a['value'], B[j][0]
    return None, None
t0=time.time(); rows=[]
for sid in selA:
    B = TB[sid]; ks = [k for k,(t,at) in enumerate(B) if abs(t-13.0)<1e-9 and any(a['code']=='glucose_finger' and a['kind']==0 for a in at)]
    if not ks: continue
    k = ks[0]
    RO = run_tf(B, kmax=k, step=stepO); RE = run_tf(B, kmax=k, step=stepE)
    dt = (B[k][0]-B[k-1][0])*3600; lx, lt = last_before(B, k, 'glucose_finger')
    age = (13.0-lt)*3600 if lx is not None else 0.0
    mo = pred_mean(RO[-1][0], RO[-1][1], dt, 'glucose_finger', lx, age); me = pred_mean(RE[-1][0], RE[-1][1], dt, 'glucose_finger', lx, age)
    qe = [float((q_I.double() @ RE[j][2]['E'].double().mean(0)) if RE[j][2]['E'].dim()>1 else (q_I.double() @ RE[j][2]['E'].double())) for j in range(k)]
    rows.append(dict(sid=sid, mo=mo, me=me, dl=me-mo, qe_last=qe[-1], qe_mean=np.mean(qe)))
print('time', time.time()-t0)
LR = pd.DataFrame(rows).set_index('sid'); LR['dtool'] = T13.d.reindex(LR.index); LR['ins']=T13.ins.reindex(LR.index)
print(LR.round(3).head(12).to_string()); print(LR[['dl','dtool','ins','qe_last']].corr().round(3)); print(LR[['dl','dtool']].describe().round(3))
```