A single step reproduces the tool's state to about 0.002 and takes about 1 ms. That makes a full token-space sensitivity map cheap.

Next run, on 64 insulin and 64 norepinephrine bundles whose next bundle is the reading about 55 minutes later:
- perturb the z token along each of the 192 directions by ±0.25;
- recompute the next-reading predicted means for glucose_finger, glucose_lab, map_arterial and map_cuff;
- store the finite-difference sensitivities.

From these I'll fit, by ridge regression on the bundle sensitivities, two token directions. One should shift glucose by 1 mg/dL per unit with little MAP change, the other the reverse. The test: the fitted direction must give a consistent shift across bundles (high fraction of variance explained) and small cross-effects. If the per-bundle sensitivities disagree in sign, a single fixed z_in column won't work and I'll need another injection point.

```python
codes4 = ['glucose_finger','glucose_lab','map_arterial','map_cuff']
def lastvals(s, T):
    g = grp[s]; g = g[g.hour < T-1e-6]; outv=[]
    for c in codes4:
        gc = g[g.event==c]
        outv.append((gc.value.iloc[-1], (T-gc.hour.iloc[-1])*3600) if len(gc) else (np.nan, 0.0))
    return outv
def preds_next(h1s, o1s, k):
    dtn = float(dts[k+1]); T = float(hr3[k+1]); lv = lastvals(sid3[k], T); n = o1s.shape[0]
    ctx = ctx_of_state(o1s.double(), h1s.double(), np.full(n, dtn))
    return np.stack([N(value_mean(ctx, [c]*n, np.full(n, lv[ci][0]), np.full(n, lv[ci][1]))[0]) for ci,c in enumerate(codes4)],1)
gap_ok = np.r_[(hr3[1:]-hr3[:-1] > 0.8) & (hr3[1:]-hr3[:-1] < 1.0) & (sid3[1:]==sid3[:-1]), False]
rng = np.random.default_rng(0)
kI = rng.choice(np.where(hI & ~hN & ~first & gap_ok)[0], 64, replace=False); kN = rng.choice(np.where(hN & ~hI & ~first & gap_ok)[0], 64, replace=False)
kAll = np.r_[kI, kN]; eps = 0.25; Jtok = np.zeros((len(kAll), 192, 4)); base4 = np.zeros((len(kAll),4))
t0=time.time()
for i,k in enumerate(kAll):
    H=[];O=[]
    for sgn in (1,-1):
        for j in range(192):
            d = torch.zeros(192, dtype=dtw); d[j]=sgn*eps; h1,o1 = one_step(k, d); H.append(h1); O.append(o1)
    h0_,o0_ = one_step(k); H.append(h0_); O.append(o0_)
    P = preds_next(torch.stack(H), torch.stack(O), k)
    Jtok[i] = (P[:192]-P[192:384])/(2*eps); base4[i] = P[384]
print('secs', round(time.time()-t0,1))
np.savez('/work/Jtok.npz', Jtok=Jtok, base4=base4, kAll=kAll)
Jm = Jtok.mean(0)
print('|mean J| per output', np.linalg.norm(Jm,axis=0).round(3), 'mean |J_k|', np.linalg.norm(Jtok,axis=1).mean(0).round(3))
```