Per-stay calls take 0.049 s per stay, so 2,500 stays factual plus perturbed take about 4 minutes. I'll redo the paired-perturbation analysis properly in one run, with three checks:
1. Sanity: the vectors must predict the current reading.
2. Dose directions in head-input space, and the minimum-variance decoder's factual error against the innovation term T.
3. How well policy-expected doses can be read from the factual head input. This is needed for the "amplify and centre" design.

```python
with open('/work/notes.md','a') as f:
    f.write("- WARNING: gen_samples(df) treats df as ONE record -> call per stay (0.05 s/stay). Runs 105/107 invalid.\n")
def gen_many(groups, stays):
    out = []
    for s in stays: out += gen_samples(groups[s], pert=False)
    return out
st = st3[:2500]
grp_p = {s: g_ for s, g_ in trp.groupby('stay_id', sort=False)}
t0 = time.time(); fsF = gen_many(grp_tr, st); fsP = gen_many(grp_p, st); print('gen time %.0fs' % (time.time()-t0))
assert [(r[0], r[1], r[2]) for r in fsF] == [(r[0], r[1], r[2]) for r in fsP]
m = np.array([r[0]=='glucose_finger' and r[2] >= 0 for r in fsF])
xF = np.stack([r[8] for r in fsF])[m]; xP = np.stack([r[8] for r in fsP])[m]
rs = np.array([six[r[1]] for r in fsF])[m]; rt = np.array([int(r[2]) for r in fsF])[m]
curv = np.array([r[4] for r in fsF])[m]; gl1 = np.array([r[5] for r in fsF])[m]
LdD = lagm(dDm, rs, rt, K); Lnu = lagm(NUm, rs, rt, K); LD = lagm(Dm, rs, rt, K)
print('rows %d | corr(gen lag1, my lag1) %.3f | ridge cur resid sd %.1f of %.1f' % (len(xF), np.corrcoef(gl1, LD[:,0])[0,1], cvres(xF, curv).std(), curv.std()))
dX = xP - xF; A = np.linalg.lstsq(LdD, dX, rcond=None)[0]
r2a = 1 - ((dX - LdD@A)**2).sum()/(dX**2).sum()
print('dX on lagged dD: R2(uncentred) %.3f | |a_k|:' % r2a, np.linalg.norm(A, axis=1)[:6].round(3), '| |x| sd per dim %.3f' % xF.std(0).mean())
Sig = np.cov(xF.T); Sig += 1e-4*np.trace(Sig)/192*np.eye(192); Si = np.linalg.inv(Sig)
Gm = A@Si@A.T; w = Si@A.T@np.linalg.solve(Gm, hins); T = Lnu@hins; yhat = xF@w; e = (yhat-yhat.mean())-(T-T.mean())
print('MVDR noise sd %.2f | sd(T) %.2f sd(w.x) %.2f err sd %.2f corr %.3f' % (np.sqrt(hins@np.linalg.solve(Gm, hins)), T.std(), yhat.std(), e.std(), np.corrcoef(yhat, T)[0,1]))
for k in range(3):
    res = cvres(xF, LD[:,k]); print('lag%d D: sd %.3f resid %.3f | sd(nu) %.3f corr(res,nu) %.3f' % (k+1, LD[:,k].std(), res.std(), Lnu[:,k].std(), np.corrcoef(res, Lnu[:,k])[0,1]))
Bv = LD@hins; res = cvres(xF, Bv); print('B: sd %.2f resid %.2f | sd(T) %.2f | err sd(res-T) %.2f' % (Bv.std(), res.std(), T.std(), (res-T-(res-T).mean()).std()))
```