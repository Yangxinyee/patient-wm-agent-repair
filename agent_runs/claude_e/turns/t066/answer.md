Readings in the same bundle do inform each other in the tool's factual pass:
- The finger prediction moves with the lab surprise (slope 0.35), and the lab prediction with the finger surprise (0.15).
- Cuff MAP responds a little to arterial MAP, and arterial does not respond to cuff.
- Glucose and MAP do not interact.

That within-bundle mechanism explains most of my head copy's remaining error. The copy is good enough to estimate how an edit changes NLL; for exact values I will check with the tool.

Next I will measure the **model's own one-step insulin sensitivity**. I regress its factual predicted finger means (fwd_3, 100 stays) on the same controls and dose-deviation lags as the true regression, and also run the regression on the observed readings of those stays. The gap between true and model coefficients, δ, is what the edit must add. I expect a model lag-1 coefficient far smaller than the true −2.0.

```python
fa=FA[FA.event=='glucose_finger'].set_index(['stay_id','hour'])
stays3=sorted(FA.stay_id.unique()); Gs={k:v.loc[stays3] for k,v in Gtu.items()}
TFs=pd.DataFrame(np.where(np.isfinite(Gs['gf'].values),tier_of(np.nan_to_num(Gs['gf'].values,nan=0)),5),index=Gs['gf'].index,columns=Gs['gf'].columns)
DEVs=Gs['ins'].fillna(0)-tab[TFs.values]
rows=[]
for t in range(2,24):
    g0=Gs['gf'][t].values; g1=Gs['gf'][t-1].values; gl0=Gs['gl'][t].values; dv=lagmat(DEVs,t,6)
    for j,s_ in enumerate(stays3):
        if (s_,float(t+1)) in fa.index and np.isfinite(g0[j]):
            r_=fa.loc[(s_,float(t+1))]; rows.append(dict(s=s_,t=t,ym=r_.predicted_mean,y=r_.observed,g0=g0[j],g1=g1[j],gl0=gl0[j],**{f'v{l}':dv[j,l-1] for l in range(1,7)}))
RM=pd.DataFrame(rows); XX=np.column_stack([design_glu(RM),RM[[f'v{l}' for l in range(1,7)]].values])
bm=np.linalg.lstsq(XX,RM.ym.values,rcond=None)[0]; bo_=np.linalg.lstsq(XX,RM.y.values,rcond=None)[0]
rm=RM.ym.values-XX@bm; se=np.sqrt(np.diag(rm.var()*np.linalg.pinv(XX.T@XX)))[-6:]
print("rows",len(RM),"| MODEL one-step dev coefs:",np.round(bm[-6:],3),"se",np.round(se,2))
print("observed (these stays):",np.round(bo_[-6:],2)," | true (train):",np.round(bv[-6:],2))
```