`view_from` returns a tuple. Its first element is q, matching Q99 to 1e-15; the others are the raw view and the 64 prior parameters.

Now the real analysis, unchanged:
- q at all 5,521 readings for the original and fwd8-edited models;
- the regression of Δq on the norepinephrine and insulin rates given 1–4 hours earlier;
- the best linear-readout SNR per unit dose² at glucose and MAP readings, against q's natural covariance.

Decision rule: an SNR of about 10 or more per unit² means a value-head readout of a z channel is viable. An SNR of about 1 or below means the channel is buried in q's natural variation.

```python
Qo8=np.array([view_from(xo[k],d)[0].ravel() for k,d in zip(kR,dtR)])
Qe8=np.array([view_from(xe[k],d)[0].ravel() for k,d in zip(kR,dtR)])
print('Qo8 vs Q99 max abs diff %.2e'%np.abs(Qo8-Q99).max())
dQ8=Qe8-Qo8
hR=R['hour'].values.astype(int); sR=R['stay_id'].values; rR=np.array([sid2i[s] for s in sR]); evR=R['event'].values
def lagd(arr,L):
    c=hR-L; return np.where(c>=0, arr[rR,np.clip(c,0,24)], 0.0)
XL=np.ones((len(hR),1)); cols_=['c']
for L in [1,2,3,4]: XL=np.c_[XL,lagd(ne_all,L)]; cols_.append('n%d'%L)
for L in [1,2,3,4]: XL=np.c_[XL,lagd(ins_all,L)]; cols_.append('i%d'%L)
print('|dq| mean %.4f, q natural sd-norm %.3f'%(np.linalg.norm(dQ8,axis=1).mean(), np.sqrt(Qo8.var(0).sum())))
for grp,msk in [('gluc',np.isin(evR,['glucose_finger','glucose_lab'])),('map',np.isin(evR,['map_arterial','map_cuff']))]:
    X_=XL[msk]; Y_=dQ8[msk]
    Bq_,*_=np.linalg.lstsq(X_,Y_,rcond=None); rs_=Y_-X_@Bq_
    print(grp,'n',msk.sum(),'R2 of dq on lagged doses %.3f'%(1-rs_.var(0).sum()/Y_.var(0).sum()))
    Sig=np.cov(Qo8[msk].T); ev_,V_=np.linalg.eigh(Sig); lam_=np.maximum(ev_,1e-6)
    for j,c in enumerate(cols_):
        if c=='c': continue
        r_=Bq_[j]; nr=np.linalg.norm(r_)
        naive=nr**2/max(r_@Sig@r_/nr**2,1e-12)
        out_=[(rid, r_@(V_@np.diag(1/(lam_+rid))@V_.T)@r_) for rid in [1e-4,1e-3]]
        print('  %s |r| %.4f naive SNR %.3f GLS SNR %s'%(c,nr,naive,' '.join('(%g: %.2f)'%o for o in out_)))
```