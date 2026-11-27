The decoder training data is in: 171,072 head rows from 900 stays × 4 arms, collected in 14 s. Of these, 70.9k are glucose rows (finger 52k, lab 19k) and 100k are MAP rows. The stored head inputs are consistent with the rollout. The prior-mean z matches exactly, and the softmax of the head logits matches the rollout's 4-draw averaged probabilities to 0.008 mean absolute difference.

Now the key feasibility test, with a train/test split by stay:
1. **Combined-target decoders:** ridge from with_code (sampled-z version, so the decoder learns to ignore prior-z noise) to the glucose target (Σa_k·d_{t−k} + Σc_k·p_{t−k} + const) and the MAP target (Σa^N_k·n_{t−k}). I report held-out R², plus the slope of the decoded arm-minus-recorded difference against the true difference. The slope is what drives the counterfactual response.
2. **Per-lag decodability:** R² for doses at lags 1–5 and policy lags, to see whether the late-lag terms are reachable.
3. **Gate feasibility:** a Fisher discriminant separating glucose rows from MAP rows. I use the within-(code, last-bin) covariance, since per-(code, last-bin) offsets can be cancelled exactly in the transition table. I report the cell-mean separation relative to the within-cell spread.

**Confirm:** held-out R² ≳ 0.8, slope ≈ 1 (within 0.2), and a separation ratio of at least ~50. A slope well below 1 or a weak gate would mean the in-head design cannot carry the oracle's effect.

```python
from sklearn.linear_model import Ridge
X1=wc1.numpy().astype(np.float64); X0=wc0.numpy().astype(np.float64)
codeT=DT['code']; siT=DT['si']; armT=DT['arm']; jT=DT['j']
glc=np.isin(codeT,[CODE['glucose_finger'],CODE['glucose_lab']]); mapr=np.isin(codeT,[CODE['map_arterial'],CODE['map_cuff']])
s_i=dI@np.array(a_i)+pI@np.array(c_i)+k_i; s_n=dN@np.array(a_n)+k_n
rngs=np.random.default_rng(5); test_st=set(rngs.choice(len(RT['stays']),180,replace=False)); te=np.array([s in test_st for s in siT])
def fit_eval(mask,y,alpha=1.0,nm=''):
    tr_=mask&~te; ts_=mask&te
    m=Ridge(alpha=alpha).fit(X1[tr_],y[tr_]); p1=m.predict(X1[ts_]); p0=m.predict(X0[ts_])
    r2=lambda p:1-np.var(y[ts_]-p)/np.var(y[ts_])
    # interventional response: differences vs arm 0 for same reading
    df_=pd.DataFrame({'si':siT[ts_],'j':jT[ts_],'a':armT[ts_],'y':y[ts_],'p':p0})
    b0=df_[df_.a==0].set_index(['si','j']); dd=df_[df_.a>0].join(b0[['y','p']],on=['si','j'],rsuffix='0').dropna()
    dy=dd.y-dd.y0; dp=dd.p-dd.p0; sl=float((dy*dp).sum()/(dy*dy).sum()); r2d=1-np.var(dp-dy)/np.var(dy)
    print(nm,'R2(z samp) %.3f R2(z mu) %.3f | diff slope %.3f diffR2 %.3f | sd y %.2f'%(r2(p1),r2(p0),sl,r2d,np.std(y[ts_])))
    return m
for a_ in [0.1,1.0]:
    mI=fit_eval(glc,s_i,a_,'GLC a=%g'%a_); mN=fit_eval(mapr,s_n,a_,'MAP a=%g'%a_)
for k in range(5):
    fit_eval(glc,dI[:,k],1.0,'ins d%d'%(k+1)); fit_eval(glc,pI[:,k],1.0,'pol p%d'%(k+1))
for k in range(3): fit_eval(mapr,dN[:,k],1.0,'ne n%d'%(k+1))
# gate: Fisher discriminant glucose vs MAP with within-(code,lb) covariance
lbT=DT['lb'].numpy(); cell=codeT*17+lbT
Xc=X1.copy(); 
for c in np.unique(cell): mm=cell==c; Xc[mm]-=X1[mm].mean(0)
Sw=np.cov(Xc[glc|mapr].T)+1e-6*np.eye(192)
dmu=X1[glc].mean(0)-X1[mapr].mean(0); wg=np.linalg.solve(Sw,dmu)
proj=X1@wg; sdw=np.sqrt(wg@Sw@wg)
cm=pd.Series(proj).groupby(cell).agg(['mean','count'])
print('gate: within-cell sd %.4g, glucose cells min %.4g, MAP cells max %.4g, ratio %.1f'%(sdw,cm['mean'][np.isin(cm.index//17,[6,10])].min(),cm['mean'][np.isin(cm.index//17,[7,5])].max(),(cm['mean'][np.isin(cm.index//17,[6,10])].min()-cm['mean'][np.isin(cm.index//17,[7,5])].max())/sdw))
```