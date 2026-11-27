`q_I` is a numpy array; I'll convert it to a float64 tensor and rerun the same comparison.

```python
qI_t = torch.as_tensor(np.asarray(q_I), dtype=torch.float64)
t0=time.time(); rows=[]
for sid in selA:
    B = TB[sid]; ks = [k for k,(t,at) in enumerate(B) if abs(t-13.0)<1e-9 and any(a['code']=='glucose_finger' and a['kind']==0 for a in at)]
    if not ks: continue
    k = ks[0]
    RO = run_tf(B, kmax=k, step=stepO); RE = run_tf(B, kmax=k, step=stepE)
    dt = (B[k][0]-B[k-1][0])*3600; lx, lt = last_before(B, k, 'glucose_finger')
    age = (13.0-lt)*3600 if lx is not None else 0.0
    mo = pred_mean(RO[-1][0], RO[-1][1], dt, 'glucose_finger', lx, age); me = pred_mean(RE[-1][0], RE[-1][1], dt, 'glucose_finger', lx, age)
    Ej = RE[-1][3]; print_shape = tuple(Ej.shape)
    qe = [float(qI_t @ (RE[j][3].double().reshape(-1,192).mean(0))) for j in range(k)]
    rows.append(dict(sid=sid, mo=mo, me=me, dl=me-mo, qe_last=qe[-1], qe_sum=np.sum(qe)))
print('time', time.time()-t0, 'E shape', print_shape)
LR = pd.DataFrame(rows).set_index('sid'); LR['dtool'] = T13.d.reindex(LR.index); LR['ins']=T13.ins.reindex(LR.index)
print(LR.round(3).head(12).to_string()); print(LR[['dl','dtool','ins','qe_last','qe_sum']].corr().round(3)); print(LR[['dl','dtool']].describe().round(3))
```