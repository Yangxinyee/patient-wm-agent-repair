The relay idea fails. The direction with the best signal-to-noise ratio only exploits degenerate, near-zero-variance directions of the head input. There the response is erratic: its heterogeneity is about 9–10 times the mean slope, so even a huge injection would carry no reliable dose signal.

The head-level decoder therefore remains the NE channel. The question is how much NE correction the factual budget allows, and whether a better decoding target helps. The rollout needs the response to a sustained ×2 level, so a decoder of the recent dose level (mean of lags 1–3) may beat a pure lag-1 decoder.

This run refits ridge decoders on the perturbed-dose samples for lag 1 and for the 3-hour level, for both drugs. On setA with exact tool residuals and bin-mean compensation, it reports E[δ²] and E[rδ] and the predicted overall NLL change at several gains c. The optimum is c* = B/A. A decoder whose cost stays near zero at the needed gain (NE about 3–5, insulin about −3) is the one to use.

```python
import pickle
from sklearn.linear_model import Ridge
SP = pickle.load(open('/work/samp_pert.pkl','rb'))
spc = np.array([str(t[0]) for t in SP]); spX = np.stack([t[8] for t in SP]); spD = np.array([[t[5],t[6],t[7]] for t in SP], dtype=float)
lastbinA = np.array([lastbin_of(r[0], r[3]) for r in rowsA])
res_tab = {}
for cd, Dm_, sig, frac, cs in [('glucose_finger', OU['ins'], 14.86, 0.343, [-2,-3,-4]), ('map_arterial', OU['ne'], 5.17, 0.261, [2,3,4,5])]:
    ms = spc==cd; mA = (codesA==cd) & np.isfinite(rA); r = rA[mA]; lb = lastbinA[mA]
    for tgt in ['lag1', 'level3']:
        y = spD[ms,0] if tgt=='lag1' else spD[ms].mean(1)
        mdl = Ridge(alpha=1.0).fit(spX[ms], y); dh = mdl.predict(XA_[mA])
        dtrue = lagU(Dm_,1)[mA] if tgt=='lag1' else np.mean([lagU(Dm_,k)[mA] for k in (1,2,3)], 0)
        x = dh - pd.Series(dh).groupby(lb).transform('mean').values
        A = (x**2).mean(); B = (r*x).mean(); res_tab[(cd,tgt)] = mdl
        print(cd[:6], tgt, 'slope %.2f errsd %.3f | A %.4f B %.4f c*=%.2f |' % (np.polyfit(dtrue, dh, 1)[0], (dh-dtrue).std(), A, B, B/A),
              ' '.join('c=%g:%.4f' % (c, frac*(c*c*A-2*c*B)/(2*sig**2)) for c in cs))
```