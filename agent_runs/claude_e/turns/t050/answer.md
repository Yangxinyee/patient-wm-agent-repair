This is the first estimate of the true one-step insulin effect, from 276k training rows with a residual sd of 12.8. The lag-1 coefficient is −2.01 per u/h, matching my earlier impulse estimate. Lag 2 is −0.74. Lags 3 to 6 come out positive, about +1.2 to +1.4.

Those positive lags have two possible explanations:
- **Rebound.** The drug's effect fades. Once the readings already show a drug-induced drop, glucose comes back up.
- **Confounding.** Earlier raw doses stand in for earlier glucose levels that my controls leave out.

To separate them I will replace the raw doses with dose deviations: the recorded dose minus the protocol's expected dose for the reading at that hour. These are random by protocol design.
- If lags 3 to 6 stay positive with deviations, rebound is real and my edit needs those positive terms.
- If they shrink toward zero, the positive values were confounding.

```python
gfall=Gtr['gf']; insall=Gtr['ins'].fillna(0)
hrs=[h for h in gfall.columns if 0<=h<=24]
gv=gfall[hrs].values.ravel(); iv=insall[hrs].values.ravel()
tf=np.where(np.isfinite(gv),tier_of(np.nan_to_num(gv,nan=0)),5)
tab=np.array([iv[tf==k].mean() for k in range(6)]); print("E[dose|tier 0..4, missing]:",np.round(tab,3))
TF=pd.DataFrame(np.where(np.isfinite(gfall.values),tier_of(np.nan_to_num(gfall.values,nan=0)),5),index=gfall.index,columns=gfall.columns)
DEV=insall - tab[TF.values]
rowsI=[]
for t in range(2,24):
    y=G['gf'][t+1].values; g0=G['gf'][t].values; g1=G['gf'][t-1].values; gl0=G['gl'][t].values
    dv=lagmat(DEV,t,6); ok=np.isfinite(y)&np.isfinite(g0)
    rowsI.append(pd.DataFrame(dict(y=y[ok],g0=g0[ok],g1=g1[ok],gl0=gl0[ok],t=t,**{f'v{l}':dv[ok,l-1] for l in range(1,7)})))
RV=pd.concat(rowsI,ignore_index=True)
XX=np.column_stack([design_glu(RV),RV[[f'v{l}' for l in range(1,7)]].values]); bv=np.linalg.lstsq(XX,RV.y.values,rcond=None)[0]
print("one-step coefs on dose DEVIATION lag1..6:",np.round(bv[-6:],3))
```