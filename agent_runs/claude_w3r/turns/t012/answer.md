Insulin's effect builds and then fades. The reduced-form response of glucose to a one-hour insulin deviation is −1.97 at 1 h, −2.23 at 2 h (the peak), −1.74 at 3 h, −1.19 at 4 h, then about −0.12 by 8 h. These numbers include some feedback from later dosing.

The rollout holds doses fixed at recorded × factor, so I need the structural impulse response with later doses held fixed. Next I will estimate it with an IV distributed-lag regression. g_t is regressed on r_{t−1..t−8}, instrumented by the random deviations x_{t−1..t−8}, controlling for the last glucose before the window. If the structural lags come out similar to the reduced form, the feedback is weak.

```python
gfl5 = np.floor(gf_last/5)*5
# E[r | last g5] over panel cols 1..24
dfp = pd.DataFrame({'g5':gfl5[:,1:].ravel(),'r':ins_r[:,1:].ravel()})
mp = dfp.groupby('g5').r.mean()
rexp = np.full_like(ins_r, np.nan); rexp[:,1:] = pd.Series(gfl5[:,1:].ravel()).map(mp).values.reshape(n,T-1)
xdev = ins_r - rexp
def iv_dl(K, yarr, rarr, xarr, ctrl_last):
    Ys=[];Rs=[];Zs=[];Cs=[];G=[]
    for t in range(K+1, T):
        y=yarr[:,t]; c=ctrl_last[:,t-K]
        R=rarr[:,t-K:t][:,::-1]; Z=xarr[:,t-K:t][:,::-1]   # lag1..lagK
        m=~np.isnan(y)&~np.isnan(c)&~np.isnan(Z).any(1)
        Ys.append(y[m]);Rs.append(R[m]);Zs.append(Z[m]);Cs.append(np.column_stack([c[m],(c[m]-160)**2/100, np.full(m.sum(),t)]));G.append(np.arange(n)[m])
    Y=np.concatenate(Ys);R=np.vstack(Rs);Z=np.vstack(Zs);C=np.vstack(Cs);g=np.concatenate(G)
    W=np.column_stack([np.ones(len(Y)),C]); Xe=np.column_stack([R,W]); Ze=np.column_stack([Z,W])
    # 2SLS
    PZ = np.linalg.lstsq(Ze, Xe, rcond=None)[0]; Xh=Ze@PZ
    b=np.linalg.lstsq(Xh, Y, rcond=None)[0]; e=Y-Xe@b
    A_=np.linalg.inv(Xh.T@Xh); U=pd.DataFrame(Xh*e[:,None]).groupby(g).sum().values; V=A_@(U.T@U)@A_
    return b[:K], np.sqrt(np.diag(V))[:K], len(Y)
b,se,nn = iv_dl(8, gf, ins_r, xdev, gf_last)
print('n',nn); print('structural IRF lag1..8:', np.round(b,3)); print('se:', np.round(se,3))
```