The shuffle tests confirm my bootstrap standard errors were honest:

| Drug | Observed ×2 readout | Shuffled sd | Two-sided p |
|---|---|---|---|
| Insulin (finger) | +0.90 | 0.50 | 0.06 |
| Norepinephrine (combined MAP) | −0.34 | 0.13 | 0.013 |

So the norepinephrine association is unlikely to be pure chance, provided my innovations really are clean. Insulin is borderline.

Those tests take the estimated innovations as given, though. One possible bias remains. Both the dose rule and the outcome model are fitted by gradient boosting, and their small approximation errors can be correlated. Because the innovation is tiny (sd 0.136), even an error product of about 0.003 could shift each lag by about 0.1 and the readout by up to about 0.4. The minimal-versus-full gap for norepinephrine (−0.20 vs −0.34) hints that modelling choices matter at roughly that level.

To check this I will use a machine-learning-free estimator built on the known rule. Within fine cells of current MAP (integer), previous rate (0.05 bands) and reading type, the expected rate change is fixed. I demean the rate change and the later MAP change within cells and regress, with stay-clustered standard errors. I then convert to the implied ×2 readout with the same persistence and weights.

- **Supports the association:** the cell estimator also gives about −0.3.
- **Points to fitting bias:** the cell estimator lands near 0.

```python
t0=time.time()
# simple cell-FE estimator for NE on rows with a MAP reading at t (contR)
sC,tC=np.nonzero(contR)
mapt=MAPpol[sC,tC]; r1c=NE0[sC,tC-1]; rc=NE[sC,tC]; typ=np.where(~np.isnan(GA[sC,tC]),0,1)
cell=pd.factorize(pd.Series(np.round(mapt).astype(int)).astype(str)+'_'+pd.Series(np.floor(r1c/0.05).astype(int)).astype(str)+'_'+pd.Series(typ).astype(str))[0]
def fe_beta(y,x,cell_,cl_):
    ok=~np.isnan(y); df=pd.DataFrame({'y':y[ok],'x':x[ok],'c':cell_[ok],'s':cl_[ok]})
    df['yd']=df.y-df.groupby('c').y.transform('mean'); df['xd']=df.x-df.groupby('c').x.transform('mean')
    b=(df.xd*df.yd).sum()/(df.xd**2).sum(); u=df.xd*(df.yd-b*df.xd); g=u.groupby(df.s).sum(); se=np.sqrt((g**2).sum())/(df.xd**2).sum()
    return b,se,df
Ls=[];Ses=[]
for k in range(1,13):
    tt=tC+k; ok=tt<=25; y=np.full(len(sC),np.nan); y[ok]=MAPc[sC[ok],tt[ok]]-MAPc[sC[ok],tC[ok]]
    b,se,_=fe_beta(y,rc,cell,sC); Ls.append(b); Ses.append(se)
Rs=[]
for j in range(12):
    tt=tC+j; ok=tt<=25; y=np.full(len(sC),np.nan); y[ok]=NE0[sC[ok],tt[ok]]
    b,se,_=fe_beta(y,rc,cell,sC); Rs.append(b)
Ls=np.array(Ls); Rs=np.array(Rs)
print('cell-FE NE: L',Ls.round(2)); print('   se',np.array(Ses).round(2)); print('   R',Rs[:5].round(2),' implied x2 readout %.3f'%((deconv(Ls,Rs)*Aw).sum()))
# bootstrap the readout over stays using per-stay sums (cell-demeaned within full sample)
print('time',round(time.time()-t0,1))
```