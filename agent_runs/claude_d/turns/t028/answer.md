Within arterial-line stays, arterial MAP is measured 95% of the time whether or not norepinephrine ran the hour before. So no weighting is needed for map_arterial outcomes. The flexible policy model explains 85% of the rate variance given the previous rate and MAP. The leftover titration noise (SD 0.13 mcg/kg/min) is a strong instrument.

Next I will run sequential g-estimation for norepinephrine → map_arterial, lags 1–12. I use arterial-line stays, the policy residual as the instrument for hours where the infusion was already running, and recorded MAP and previous rate as controls. A rising or persistent γ_k would mean the effect builds with sustained infusion. A one-lag spike would mean an immediate, transient effect.

```python
Kn=12; gam=np.zeros(Kn+1)
A_=art_stay
for k in range(1,Kn+1):
    Yt=[];E=[];A=[];C=[];
    for t in range(k, H):
        s=t-k
        obs = A_ & ~np.isnan(MA[:,t])
        yt = MA[:,t] - sum(gam[j]*NOR[:,t-j] for j in range(1,k))
        sel = obs & cont[:,s]
        Yt.append(yt[sel]); E.append(EPSN[sel,s]); A.append(NOR[sel,s]); C.append(np.c_[np.nan_to_num(Mdec[sel,s],nan=70), np.isnan(Mdec[sel,s]), NP[sel,s], NP[sel,s]**2, np.nan_to_num(Mdec[sel,s],nan=70)**2/100, pred_s := np.zeros(sel.sum())])
    Yt=np.concatenate(Yt);E=np.concatenate(E);A=np.concatenate(A);C=np.concatenate(C)
    Xc = np.c_[np.ones(len(Yt)), C[:,:5]]
    coef=np.linalg.lstsq(Xc, Yt, rcond=None)[0]; r = Yt - Xc@coef
    gam[k] = np.sum(r*E)/np.sum(A*E)
    u = r - gam[k]*(A - A.mean())
    se = np.sqrt(np.sum((E*u)**2))/abs(np.sum(A*E))
    print(k, len(Yt), 'gamma', round(gam[k],3), 'se', round(se,3))
print('sum', gam.sum())
```