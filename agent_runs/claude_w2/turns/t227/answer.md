The gate is easy, but the dose signal is the binding constraint.
- **Gate:** the glucose-vs-MAP discriminant separates the cell means by about 600 within-cell standard deviations.
- **Lag-1 dose:** the head input encodes it only weakly in the causal sense. The decoder fits the recorded dose with R² 0.65, but its response to an actual dose change is only 0.36 per unit. The fit leans on glucose-level proxies.
- **Lags 2–5:** the response is essentially zero (0.08, 0.02, 0.02, 0.02), although the presence of insulin is decodable (R² about 0.4).

This fits the architecture. Only the 'START' atom carries the dose, and it enters through u into the fast-decaying h (a_bar about 0.4 per bundle). The long memory holds only END and measurement atoms, which carry no dose. So the model can only know how much was given at lag 1, and only whether insulin ran at earlier lags.

So the full lag 1–5 correction is not reachable from the head. The workable design is a lag-1 deviation channel: s = G1·(d̂1 − p̂1) + Σγ_k·p_k + const. Here γ are the residual's net policy-level coefficients, which are unaffected by dropping the randomized e-lags. G1 would be calibrated in rollouts, possibly beyond the factual −1.27, to stand in for the missing lag-2+ terms. Overshooting G1 on e1 is cheap in NLL (sd(e1) is small).

Next I measure the attainable decoder. I fit stacked ridge regressions on factual rows (target d1) plus arm-difference rows (target Δd1), with weight μ on the differences. For each μ I report the causal slope and the factual error sd on held-out stays, for insulin d1 and norepinephrine n1. I also get γ and β from the 2000-stay residual table.

**Confirm:** slope reaching about 0.9 with error sd below about 1.5 U/h equivalent. **Refute:** slope stuck well below 0.7, which would push me toward amplifying the dose signal in the encoder instead.

```python
# gamma: resid ~ p lags (+const) ; beta: d1 ~ p lags (gf rows, 2000-stay table)
Mi=M3[(M3.code==6)&(M3.h>=2)]
tg,_,_=ols_tab(Mi,'res',['p%d'%k for k in [1,2,3,4,5]]); tb,_,_=ols_tab(Mi,'d1',['p%d'%k for k in [1,2,3,4,5]])
te1,_,_=ols_tab(Mi,'res',['e1','e2','e3']+['p%d'%k for k in [1,2,3,4,5]])
print(pd.concat([tg.reset_index(),tb.reset_index()],axis=1).to_string()); print(te1.T.to_string())
# stacked difference-trained ridge decoders
def pair_diffs(mask):
    df_=pd.DataFrame({'r':np.arange(len(siT)),'si':siT,'j':jT,'a':armT})[mask]
    b0=df_[df_.a==0].set_index(['si','j'])['r']; dd=df_[df_.a>0].join(b0.rename('r0'),on=['si','j']).dropna()
    return dd.r.values.astype(int), dd.r0.values.astype(int)
def stacked(mask,y,mus,alpha=1.0,nm=''):
    f_tr=np.where(mask&~te&(armT==0))[0]; f_ts=np.where(mask&te&(armT==0))[0]
    r1,r0=pair_diffs(mask); trp=~te[r1]; tsp=te[r1]
    Xd=X0[r1]-X0[r0]; yd=y[r1]-y[r0]
    out=[]
    for mu in mus:
        Xs=np.vstack([X1[f_tr],Xd[trp]]); ys=np.r_[y[f_tr],yd[trp]]; sw=np.r_[np.ones(len(f_tr)),mu*np.ones(trp.sum())]
        m=Ridge(alpha=alpha).fit(Xs,ys,sample_weight=sw)
        pd_=Xd[tsp]@m.coef_; sl=float((yd[tsp]*pd_).sum()/(yd[tsp]**2).sum())
        ef=y[f_ts]-m.predict(X1[f_ts]); out.append((mu,round(sl,3),round(float(np.std(ef)),3),round(float(np.std(pd_-sl*yd[tsp])),3),round(float(np.linalg.norm(m.coef_)),2)))
    print(nm,'sd y factual %.2f'%np.std(y[f_ts]),' (mu, causal slope, factual err sd, diff noise sd, |w|):'); [print('   ',o) for o in out]
    return m
mus=[0,1,10,100,1000]
_=stacked(glc,dI[:,0],mus,1.0,'INS d1')
_=stacked(glc,dI[:,1],[10,1000],1.0,'INS d2')
_=stacked(mapr,dN[:,0],mus,1.0,'NE n1')
```