Run 91 checks out:
- The recomputed `with_code` inputs reproduce the hidden activations exactly (max diff 3e-6).
- panel.pkl holds the full hourly training panel: stay ids, hours −1..24, and one 30000×26 array per event. Readings sit at integer hours; insulin and norepinephrine rates sit at their start hour.

Next I'll build a proper insulin policy expectation from the 30,000 training stays. It is the cell mean of the rate given the band of the last glucose_finger, whether that reading is fresh (taken this hour), and early hours (≤2). From it I get the deviations dev1 (current hour's dose) and dev2 (previous hour's) for each glucose_finger row of fwd_3.

On those rows I will then:
- fit ridge probes on `with_code` for dev1 and dev2, with grouped CV;
- measure each probe's slope k on the true deviation;
- re-estimate the model's own partial responses γ1 and γ2.

**What I expect:** k1 ≈ 0.4; k2 decodable only if the END atom carries the previous rate; γ1 ≈ +0.3. **What would break the plan:** k1 far below 0.3, because the needed rescaling 1/k would then amplify factual noise too much.

```python
sid, hrs, PA = PN; sidx = {s:i for i,s in enumerate(sid)}
Gf = PA['glucose_finger']; Ins = np.nan_to_num(PA['insulin'], nan=0.0)
glc = pd.DataFrame(Gf).ffill(axis=1).values; fresh = ~np.isnan(Gf)
Bd = np.where(np.isnan(glc), -1, np.searchsorted([150,180,220,260,300], np.nan_to_num(glc), side='right'))
early = (hrs[None,:] <= 2) & np.ones_like(Bd, bool)
cell = (Bd+1)*4 + fresh.astype(int)*2 + early.astype(int)
Mcell = np.zeros_like(Ins); tab = {}
for c in np.unique(cell[:,1:]):
    mk = (cell==c); mk[:,0] = False; tab[c] = (Ins[mk].mean(), mk.sum(), (Ins[mk]>0).mean()); Mcell[mk] = tab[c][0]
DEV = Ins - Mcell
print('policy cells (band, fresh, early): mean rate, n, P(on)')
for c,(m_,n_,p_) in sorted(tab.items()):
    if n_>500: print(((c//4)-1, (c%4)//2, c%2), round(m_,3), n_, round(p_,3))
gfm = (MT.event=='glucose_finger').values & (MT.hour.values>=1)
rows = np.where(gfm)[0]; si = np.array([sidx[s] for s in MT.stay_id.values[rows]]); H = MT.hour.values[rows].astype(int)
d1 = DEV[si, H]; d2 = DEV[si, H-1]; r1 = Ins[si, H]; r2 = Ins[si, H-1]
Xg = XW[rows]; grp_g = MT.stay_id.values[rows]
def cvfit(Xm, y, g):
    yp = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xm, y, g): yp[te] = RidgeCV(alphas=(1,10,100,1000)).fit(Xm[tr], y[tr]).predict(Xm[te])
    k = np.polyfit(y, yp, 1)[0]; return yp, round(1-np.mean((y-yp)**2)/np.var(y),3), round(k,3)
for nm, y in [('dev1',d1),('dev2',d2),('r1',r1),('r2',r2)]:
    _, r2_, k_ = cvfit(Xg, y, grp_g); print(nm, 'sd', y.std().round(3), 'CV R2', r2_, 'slope k', k_)
ctrl = np.column_stack([np.eye(17)[[fbin('glucose_finger','value',g) if g==g else 0 for g in glc[si,H]]], np.eye(17)[[fbin('glucose_finger','value',g) if g==g else 0 for g in glc[si,H-1]]], np.nan_to_num(glc[si,H]), H])
for nm, yy in [('model pm', MT.pm.values[rows]), ('records obs', MT.obs.values[rows])]:
    lr = LinearRegression().fit(np.column_stack([d1, d2, ctrl]), yy); print(nm, 'slopes on dev1, dev2:', lr.coef_[:2].round(3))
```