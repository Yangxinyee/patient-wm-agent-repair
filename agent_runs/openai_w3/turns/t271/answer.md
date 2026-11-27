The first dose extraction counted infusion END records as fresh doses, so that fit is not trustworthy. I’ll exclude END records, refit with balanced weights across stopping, halving, doubling and tripling, then install a reversible gain test. It is only a candidate: factual NLL and paired rollouts must confirm any improvement.

```python
repair_doses243 = np.zeros((len(idx81),2),dtype=np.float64)
for j243,i243 in enumerate(np.asarray(idx81)):
    for row243,end243 in r40_info[int(i243)]:
        if end243 or row243.event not in ("insulin","norepinephrine"):
            continue
        rr243 = float(row243.rate)
        if np.isfinite(rr243):
            repair_doses243[j243,0 if row243.event=="insulin" else 1] += rr243
print("Correct dose maxima", repair_doses243.max(axis=0))
print("Difference from cached record doses",
      np.max(np.abs(repair_doses243-np.asarray(rate88))))

repair_X243 = np.asarray(X220,dtype=np.float64)
repair_tr243 = np.asarray(tr220,dtype=bool)
repair_va243 = np.asarray(va220,dtype=bool)
ntr243 = int(repair_tr243.sum())
repair_cov243 = 0.5*(repair_X243[repair_tr243].T @
                     repair_X243[repair_tr243])/ntr243
repair_rhs243 = np.zeros((769,2))
repair_targets243 = {}
for key243,dx243 in contrast_features220.items():
    drug243,mult243 = key243
    yy243 = np.zeros((len(idx81),2))
    yy243[:,drug243] = (mult243-1.0)*repair_doses243[:,drug243]
    yy243 = np.repeat(yy243,4,axis=0)
    repair_targets243[key243] = yy243
    dt243 = np.asarray(dx243,dtype=np.float64)[repair_tr243]
    yt243 = yy243[repair_tr243]
    wt243 = 1.0/(len(contrast_features220)*(mult243-1.0)**2*ntr243)
    repair_cov243 += wt243*(dt243.T @ dt243)
    repair_rhs243 += wt243*(dt243.T @ yt243)
repair_cov243 += np.eye(769)*1e-4
repair_beta243 = np.linalg.solve(repair_cov243,repair_rhs243)
print("Held-out factual innovation RMS",
      np.sqrt(np.mean((repair_X243[repair_va243]@repair_beta243)**2,axis=0)))
for key243,dx243 in contrast_features220.items():
    pr243 = np.asarray(dx243)[repair_va243]@repair_beta243
    yy243 = repair_targets243[key243][repair_va243]
    drug243 = key243[0]
    nz243 = np.abs(yy243[:,drug243])>1e-8
    sl243 = np.sum(pr243[nz243,drug243]*yy243[nz243,drug243])/np.sum(yy243[nz243,drug243]**2)
    print(key243, "slope", round(float(sl243),3),
          "error RMS", np.round(np.sqrt(np.mean((pr243-yy243)**2,axis=0)),4))

checkpoint243 = {n:as_numpy(load(n)).copy() for n in edited()}
repair_coef243 = repair_beta243[:-1]/np.asarray(hs220)[:,None]
repair_intercept243 = repair_beta243[-1]-np.asarray(hm220)@repair_coef243
repair_gain_delta243 = np.array([-0.35,1.0])
repair_dirs243 = np.asarray(dd193)*repair_gain_delta243[None,:]
candidate243 = {n:v.copy() for n,v in checkpoint243.items()}
candidate243["core.prelude.0.ffn.2.weight"] += (
    repair_dirs243@repair_coef243.T).astype(np.float32)
candidate243["core.prelude.0.ffn.2.bias"] += (
    repair_dirs243@repair_intercept243).astype(np.float32)
assert all(np.isfinite(v).all() for v in candidate243.values())
for n243,v243 in candidate243.items():
    save_edit(n243,tensor153(v243))
print("Installed candidate243; gain increments",repair_gain_delta243)
print("FFN adjustment norm",
      np.linalg.norm(repair_dirs243@repair_coef243.T))
print("Validation stay IDs",ids221[:5],"...",ids221[-1])
with open("/work/notes.md","a") as f:
    f.write("\nPython run 243; forwards 16. Corrected physical-dose extraction to exclude END records and verified against rate88. Fitted balanced 0/.5/2/3 treatment-contrast decoder with whole-stay holdout. Installed reversible rank-two FFN candidate243 (insulin gain increment -.35, NE +1.0); checkpoint243 retains validated checkpoint222. Next: authoritative factual and paired-rollout validation.\n")
```