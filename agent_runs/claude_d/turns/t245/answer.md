The gates look clean. Gated rows (arterial MAP with norepinephrine in the previous hour) have G ≥ 1.61, and 99% of them are above 1.82. Every other category has a 99th percentile at or below 1.35. The only exception is the top 0.1% of arterial rows without norepinephrine, so the AND-gate leaks almost nowhere.

The z-augmented decoder of the norepinephrine dose surprise improves as ridge alpha shrinks: cvR2 0.20 with slope k 0.24 at alpha 0.3. Cuff rows show a similar factual deficit (2.3). I leave cuff rows untouched, since the tilt cannot treat them correctly.

Next I will build the channel on units 54 and 9. The build:
- removes their old contribution and compensates in the bias;
- uses gate scale 6 and a small tilt (lam 1, t = (centre − 75)/10);
- rounds everything to bf16;
- sizes the decoder gain from the gated rows' predictive variance.

I will then evaluate it offline for gains of 3, 6 and 9 mmHg per unit of surprise. For each gain I check:
- in-sample and held-out-stay NLL change;
- whether the realized mean shift per unit of surprise matches the intended gain;
- the bf16 bias.

This will show how much gain the 0.005 budget allows on top of E3b's insulin cost.

```python
import torch
from scipy.special import erf
def bfr_(x): return torch.tensor(np.asarray(x, dtype=np.float32)).to(torch.bfloat16).float().numpy()
def gel(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
cenA = np.asarray(VOCAB['value_bin_centres']['map_arterial'], dtype=np.float64)
tA = np.r_[0.0, (cenA-75)/10] if len(cenA)==16 else (cenA-75)/10
W1o_,b1o_,W2o_,b2o_ = [np.asarray(load_original('attr_heads.value_bin.%s'%s)) for s in ['0.weight','0.bias','2.weight','2.bias']]
r0 = eval_E(W1o_,b1o_,W2o_,b2o_); rE = eval_E(W1c,b1c,W2c,b2c)
print('chk', len(cenA), round(np.corrcoef(r0[0], RD2.nll0)[0,1],4), round(np.corrcoef(r0[1], RD2.pred0)[0,1],4), 'E3b dAll', round(float((rE[0]-r0[0]).mean()),5))
Pg = P0[gr,1:]; mg = Pg@cenA; Vref = float((Pg@cenA**2 - mg**2).mean()); print('Vref', round(Vref,1))
mact = gel(WCd@W1c.T + b1c)[:, :, [54,9]].mean((0,1))
def build_ne(C_ne, wd, bd, k, units=(54,9), kap=6.0, lam=1.0):
    ia, ib = units; W1n,b1n,W2n,b2n = W1c.copy(),b1c.copy(),W2c.copy(),b2c.copy()
    b2n = b2n + W2c[:,ia]*mact[0] + W2c[:,ib]*mact[1]
    sig = C_ne*10.0/(k*lam*Vref); wg = kap*(wf+wp); bg = kap*(bf+bp-1.5)
    W1n[ib] = wg; b1n[ib] = bg; W1n[ia] = wg + sig*wd; b1n[ia] = bg + sig*bd
    W2n[:,ia] = lam*tA; W2n[:,ib] = -lam*tA
    return [bfr_(x) for x in (W1n,b1n,W2n,b2n)]
yD = Dn[gr]; best=None
for a in [0.1, 0.03]:
    p = np.zeros(len(gr))
    for tr, te in GroupKFold(5).split(gr, groups=sid[gr]):
        w, b = ridge_fit(WCd[:, gr[tr], :].reshape(-1,192), np.tile(yD[tr], nd), a); p[te] = (WCd[:, gr[te], :]@w + b).mean(0)
    r2 = 1-((yD-p)**2).mean()/yD.var(); kk = np.cov(p,yD)[0,1]/yD.var(); print('dec a', a, round(r2,3), 'k', round(kk,3))
    if best is None or r2>best[1]: best=(a,r2,kk)
a_d, _, k_d = best
wd, bd = ridge_fit(WCd[:, gr, :].reshape(-1,192), np.tile(yD, nd), a_d)
Dh = (WCd[:, gr, :]@wd + bd).mean(0)/k_d
noA = isA&(ne1<=0)
for C_ne in [3.0, 6.0, 9.0]:
    Wn = build_ne(C_ne, wd, bd, k_d); r = eval_E(*Wn)
    dn = r[0]-r0[0]; sl = np.polyfit(Dh, (r[1]-rE[1])[gr], 1)[0]
    rb = eval_E(*Wn, dt=torch.bfloat16)
    print('C', C_ne, 'dAll', round(float(dn.mean()),5), 'gated', round(float(dn[gr].mean()),4), 'A&no', round(float(dn[noA].mean()),4), 'slope', round(sl,2), 'bf16 bias', round(float((rb[1]-r[1])[gr].mean()),3))
```