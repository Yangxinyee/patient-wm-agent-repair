Run 188 failed on indexing. Some drug-bundle rows have an hour index of 25, past the 0–24 columns of the per-hour dose table, so I need to drop them.

The run also showed that ctx sometimes follows directly from the previous drug bundle (dt = 3600 s) when no reading bundle sits in between. Those rows are valid; ctx then just has a different gap.

I will rerun the same analysis with hours ≥ 25 excluded. Same question and same criteria as before: is the ridge R² of rate on ctx close to what the policy itself explains? That would mean a linear expected-dose term in the writer suffices.

```python
keep=np.floor(HR).astype(int)<=24
Xp,DT,SID,HR,Z,NA=Xp[keep],DT[keep],SID[keep],HR[keep],Z[keep],NA[keep]
CT=np.array([ctx_from(Xp[k],DT[k]) for k in range(len(DT))])
hcol=np.floor(HR).astype(int)
ri=np.array([sid2i[s] for s in SID])
Dins=ins_all[ri,hcol]; Dne=ne_all[ri,hcol]
err=[]
for k in range(0,len(DT),len(DT)//200):
    zz_=zmean(CT[k],E_EMPTY,u_rec(Dins[k],Dne[k]))
    err.append(np.abs(np.asarray(zz_).ravel()[:32]-Z[k]).mean())
print('rows',len(DT),'z recon mean abs err', np.mean(err), 'z sd', Z.std(0).mean())
gkf=GroupKFold(5)
def cvr2(X,y,al):
    p=np.zeros_like(y)
    for tr_,te_ in gkf.split(X,y,SID):
        m=Ridge(alpha=al).fit(X[tr_],y[tr_]); p[te_]=m.predict(X[te_])
    return 1-((y-p)**2).mean()/y.var(), np.sqrt(((y-p)**2).mean()), p
for nm,y in [('ins',Dins),('ne',Dne)]:
    for al in [0.1,1,10]:
        r2,rm,_=cvr2(CT,y,al); print(nm,'alpha',al,'cvR2 %.3f rmse %.3f'%(r2,rm),'sd %.3f'%y.std())
gcol=np.array([list(hrs).index(h) for h in hcol]); g_=glast[ri,gcol]
base_=np.where(g_<180,1,np.where(g_<220,2,np.where(g_<260,3,4.3)))
on_=Dins>0
print('ins on frac %.3f'%on_.mean(),' policy resid sd (on rows): %.3f'%np.std((Dins-base_)[on_]), 'n on', on_.sum())
_,_,pI=cvr2(CT,Dins,1.0); print('ctx-decoder resid sd on on-rows: %.3f'%np.std((Dins-pI)[on_]), ' off rows: %.3f'%np.std((Dins-pI)[~on_]))
np.savez('/work/drugctx.npz',CT=CT,SID=SID,HR=HR,Dins=Dins,Dne=Dne,NA=NA,Z=Z)
```