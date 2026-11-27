Insulin's effect is very homogeneous. At lag 1 it is -1.92 to -2.03 in every subgroup (norepinephrine on/off, arterial line, early/late, glucose band); lag 2 is similar. One linear impulse response for insulin should suffice.

Norepinephrine shows some heterogeneity. The lag-1 effect falls from about 3.6 per unit at previous rates below 0.6 to 3.1–3.2 at rates of 0.6 and above. It is also 3.18 at MAP below 65 versus 3.74 at 65 and above. That could be mild saturation in the rate, or lower sensitivity in sicker, hypotensive patients, who also run higher rates. The difference matters for extrapolating x2 and x3 beyond the observed cap of 1.0.

Next I'll cross-tabulate the lag-1 norepinephrine effect by previous-rate bin and MAP level jointly to separate the two explanations. I'll also check the cross-effects (norepinephrine → glucose, insulin → MAP). If the decline persists within each MAP stratum, it's saturation in the rate. If it vanishes, it's state heterogeneity.

```python
m0 = Bn['y1'].notna().values
ryn1 = np.full(len(Bn), np.nan); ryn1[m0] = crossfit_resid(Xn[m0], Bn['y1'].values[m0])
def slope(m, rd, ry):
    b = (rd[m]@ry[m])/(rd[m]@rd[m]); e = ry[m]-b*rd[m]; se = np.sqrt((e**2*rd[m]**2).sum())/(rd[m]@rd[m]); return round(b,2), round(se,2), int(m.sum())
for mlo,mhi in [(0,60),(60,65),(65,75),(75,200)]:
    row=[]
    for lo,hi in [(0,.3),(.3,.6),(.6,.9),(.9,1.01)]:
        m = m0 & (Bn.rp>=lo).values & (Bn.rp<hi).values & (Bn.mlast>=mlo).values & (Bn.mlast<mhi).values
        row.append(slope(m, rdn, ryn1))
    print("MAP",mlo,mhi, row)
# cross effects: norepi shock -> glucose finger at t+1,t+2 ; insulin shock -> MAP arterial
for k in [1,2,3]:
    Bn[f'g{k}'] = np.nan
Bn_g = []
for k in [1,2,3]:
    yy = np.array([gbest[s, col(t)+k] for s,t in zip(Bn.stay, Bn.t)])
    m = np.isfinite(yy); ry = np.full(len(Bn), np.nan); ry[m] = crossfit_resid(Xn[m], yy[m])
    Bn_g.append(slope(m, rdn, ry))
print("norepi -> glucose lags1-3:", Bn_g)
Bi_m=[]
for k in [1,2,3]:
    yy = np.array([mlast[s, col(t)+k] if np.isfinite(ma[s,col(t)+k]) or np.isfinite(mc[s,col(t)+k]) else np.nan for s,t in zip(Bi.stay, Bi.t)])
    m = np.isfinite(yy); ry = np.full(len(Bi), np.nan); ry[m] = crossfit_resid(Xi[m], yy[m])
    Bi_m.append(slope(m, rdi, ry))
print("insulin -> MAP lags1-3:", Bi_m)
```