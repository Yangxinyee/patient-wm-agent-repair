The metadata gives me what I need. Each reading row in R (99 stays) stores the previous state row k and the gap dt, and fwd8 uses the same 99 stays in the same row order. So `qrows(states, R.k, R.dt)` rebuilds q at every reading for both the original and the fwd8-edited model.

I first check the rebuild against the stored Q99. Then I measure, at glucose and MAP readings:
- how the change in q responds to the norepinephrine and insulin rates given 1–4 hours earlier: response vectors and the linear R²;
- the key quantity, the best achievable signal-to-noise of a linear readout. That is r'Σ⁻¹r per unit dose², with Σ the natural covariance of q at those readings.

Reading the result:
- If the best-SNR readout of the lag-1 dose has noise SD well below the innovation SD per unit gain (SNR above about 10 per unit²), a value-head readout of a z channel can work.
- If the SNR is around 1 or below, the channel signal is buried in q's natural variation, and I must strengthen the write or change the route.

```python
kR=R['k'].values.astype(int); dtR=R['dt'].values
Qo8=qrows(xo,kR,dtR); Qe8=qrows(xe,kR,dtR)
print('Qo8 vs Q99 max abs diff %.2e'%np.abs(Qo8-Q99).max())
dQ8=Qe8-Qo8
hR=R['hour'].values.astype(int); sR=R['stay_id'].values; rR=np.array([sid2i[s] for s in sR]); evR=R['event'].values
def lagd(arr,L):
    c=hR-L; return np.where(c>=0, arr[rR,np.clip(c,0,24)], 0.0)
XL=np.c_[np.ones(len(hR))]+0
cols_=['c']
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
        for rid in [1e-4,1e-3]:
            Si=V_@np.diag(1/(lam_+rid))@V_.T
            print('  %s |r| %.4f  naive SNR %.3f  GLS SNR(ridge %g) %.2f'%(c,nr,naive,rid,r_@Si@r_))
```