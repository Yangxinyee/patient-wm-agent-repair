The ×3 prelude test does not even triple the direct response: at lag 1 insulin ×2 goes from −0.43 to only −0.61. The NE response is negative for both ×2 and ×½ at every lag. That is a symmetric "surprise" effect, with essentially no causal NE component in the model.

So the drug signal enters the state weakly and non-linearly. In h-space it is buried (SNR 0.17, possibly overfit).

A potent channel that does not change the state dynamics is head-side: z_head applied to the prior sample, or the value head's input. Any head-side remap needs a source that reads the drug imprint selectively from the state at glucose-query time. This run measures that:
- regress out at the bundle just before each glucose_finger reading on the previous hour's dose noise;
- take the optimal direction;
- report its generalised SNR in-sample and out-of-sample (split by stay).

If the out-of-sample SNR is well below 1, head-side remaps from state directions are ruled out.

```python
gfr=t[t.event=='glucose_finger'][['stay_id','hour','value']]
rows=[];E1=[];T1=[]
insd=ins.set_index(['stay_id','ih'])
sid_idx={s:np.where(sid==s)[0] for s in ids}
for s,h,v in gfr.itertuples(index=False):
    ii=sid_idx[s]; prior=ii[hr[ii]<h-1e-6]
    if len(prior)==0 or h<1: continue
    k=prior[-1]; key2=(s,h-1)
    if key2 in insd.index:
        e=insd.loc[key2,'eps']; e=e.iloc[0] if hasattr(e,'iloc') else e
        tr=insd.loc[key2,'tier']; tr=tr.iloc[0] if hasattr(tr,'iloc') else tr
        rows.append(k);E1.append(e);T1.append(tr)
rows=np.array(rows);E1=np.array(E1);T1=np.array(T1)
O=OUT[rows]; print(len(rows))
st=np.array([sid[r] for r in rows]); half=np.isin(st,ids[::2])
def fitd(m):
    A=np.c_[np.ones(m.sum()),pd.get_dummies(T1[m]).values[:,1:],E1[m]]
    return np.linalg.lstsq(A,O[m],rcond=None)[0][-1]
dA=fitd(half); dB=fitd(~half)
for nm,d1,m in [('A->B',dA,~half),('B->A',dB,half)]:
    C=np.cov(O[m].T); v=np.linalg.solve(C+1e-3*np.trace(C)/192*np.eye(192),d1)
    proj=O[m]@v; dd=np.polyfit(E1[m],proj,1)[0]
    print(nm,'oos SNR',round(dd**2*E1[m].var()/proj.var(),4), 'corr(dA,dB)',round(np.corrcoef(dA,dB)[0,1],3))
```