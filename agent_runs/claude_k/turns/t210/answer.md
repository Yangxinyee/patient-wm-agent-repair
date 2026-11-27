`qrows` expects a different argument layout than I assumed. Rather than guess at it, I will use `view_from(prev_row, dt)`, which my notes record as reproducing Q99 exactly for these 99 stays. I first check one row against Q99, then apply it row by row for both fwd8 models. The analysis is unchanged:
- the response of Δq to lagged rates;
- the best-readout SNR per unit dose² at glucose and MAP readings.

The decision rule is the same as before: an SNR of about 10 or more per unit² makes a value-head readout viable, and an SNR of about 1 means the z channel cannot be read without amplifying noise.

```python
q0=np.asarray(view_from(xo[kR[0]],dtR[0])); print('view_from shape',q0.shape,' diff vs Q99[0] %.2e'%np.abs(q0.ravel()[:192]-Q99[0]).max())
Qo8=np.array([np.asarray(view_from(xo[k],d)).ravel()[:192] for k,d in zip(kR,dtR)])
Qe8=np.array([np.asarray(view_from(xe[k],d)).ravel()[:192] for k,d in zip(kR,dtR)])
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
        out_=[ (rid, r_@(V_@np.diag(1/(lam_+rid))@V_.T)@r_) for rid in [1e-4,1e-3]]
        print('  %s |r| %.4f naive SNR %.3f GLS SNR %s'%(c,nr,naive,' '.join('(%g: %.2f)'%o for o in out_)))
```