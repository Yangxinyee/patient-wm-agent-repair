The test refutes the premise of a pure readout edit, at least through `out`. Before each finger reading, the state predicts the recorded insulin rate fairly well (CV R² about 0.75–0.83). That is almost all explained by the glucose reading itself, because the dosing rule alone gives R² 0.81. The part of the dose the rule doesn't explain, the innovation that drives the causal effect, is barely in the state:
- R² is 0.10 from `out` and 0.34 from the full state at lag 1.
- It is about 0.05 at lags 2–3.

So the model mostly discards the dose information. That is consistent with its effects being about 10x too small. A head-only edit would have nothing to read.

The dose has to be carried into the state upstream. One candidate: the latent z posterior is computed directly from the bundle's action encoding u_k and is meant to capture what is new in a bundle. So it may encode the innovation strongly even though the recurrent state doesn't. I'll check this at the insulin bundles themselves.
- High R² from z would give an injection point through `z_in`.
- Low R² means the loss happens already in the action encoder.

```python
rowsB=[]
for s_ in np.unique(sid_):
    ii=np.where((sid_==s_)&(z['has_insulin']==1))[0]
    for j in ii:
        t=int(np.floor(hr_[j]+1e-6)); rowsB.append((s_,t,j))
RB=pd.DataFrame(rowsB,columns=['sid','t','row'])
RB['D0']=[Ptu2.D.get((s_,t),np.nan) for s_,t in zip(RB.sid,RB.t)]
RB['nu0']=[Ptu2.nu.get((s_,t),np.nan) for s_,t in zip(RB.sid,RB.t)]
RB['G0']=[Ptu2.Gany.get((s_,t),np.nan) for s_,t in zip(RB.sid,RB.t)]
RB=RB.dropna()
XB=X_[RB.row.values]; XB=(XB-XB.mean(0))/(XB.std(0)+1e-6)
gkf=GroupKFold(5)
def cvr2g(Xm,y,grp):
    pr=np.zeros_like(y)
    for a,b in gkf.split(Xm,y,grp):
        m=RidgeCV(alphas=np.logspace(-2,4,13)).fit(Xm[a],y[a]); pr[b]=m.predict(Xm[b])
    return 1-np.mean((y-pr)**2)/np.var(y)
print('n insulin bundles',len(RB))
for c in ['D0','nu0']:
    y=RB[c].values.astype(float)
    print(c,'R2 z',round(cvr2g(XB[:,960:],y,RB.sid),3),'R2 out',round(cvr2g(XB[:,:192],y,RB.sid),3),'R2 h',round(cvr2g(XB[:,192:960],y,RB.sid),3))
# z: partial on nu controlling for G-bins
Xc=np.column_stack([XB[:,960:], pd.get_dummies((RB.G0//20).clip(5,16)).values])
print('nu0 R2 z+Gbins',round(cvr2g(Xc.astype(float),RB.nu0.values,RB.sid),3))
```