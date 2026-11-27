`view_from` returns (q, ctx, prior parameters). Its first element matches `Q99` exactly, and it accepts batches.

Now the propagation analysis on fwd_8:
1. Compute q at all 5521 readings for both models, and Δq.
2. Regress Δq on the z18 and z28 shifts of the 12 preceding bundles, plus an intercept. If R² is high, the state's response is linear and adds up across bundles.
3. Per bundle lag, compare the response vector J with the factual spread of q, using the signal-to-noise J′Σ⁻¹J. That tells me how readable each lag is.

A high R² and large per-unit SNR at the first lags would make a value-head reader viable at a small factual cost. A low SNR means the writer gain has to grow, or this route fails.

```python
Qo=np.asarray(view_from(X8o[kidx], dts99)[0]); Qe=np.asarray(view_from(X8e[kidx], dts99)[0]); dQ=Qe-Qo
print('err vs Q99',np.round(np.abs(Qo-Q99).max(),5),'| |dq| mean',np.round(np.linalg.norm(dQ,axis=1).mean(),4),'| sqrt tot var q',np.round(np.sqrt(Qo.var(0).sum()),3))
Lg=12
Z18=np.zeros((5521,Lg)); Z28=np.zeros((5521,Lg)); valid=np.ones(5521,bool)
for r__ in range(5521):
    k__=kidx[r__]
    for l__ in range(Lg):
        kk_=k__-l__
        if kk_<0 or sid8[kk_]!=sid8[k__]: break
        Z18[r__,l__]=dz[kk_,18]; Z28[r__,l__]=dz[kk_,28]
Xreg=np.c_[np.ones(5521),Z18,Z28]
B_,_,_,_=np.linalg.lstsq(Xreg,dQ,rcond=None); pred_=Xreg@B_
print('R2 of dq on lagged dz (pooled over dims)',np.round(1-((dQ-pred_)**2).sum()/((dQ-dQ.mean(0))**2).sum(),4))
Sq=np.cov(Qo.T)+1e-6*np.eye(192); Sqi=np.linalg.inv(Sq)
for nm_,off_ in [('z18',1),('z28',1+Lg)]:
    J_=B_[off_:off_+Lg]
    print(nm_,'|J| by bundle lag',np.round(np.linalg.norm(J_,axis=1),4))
    print(nm_,'SNR J S^-1 J by lag',np.round(np.einsum('ld,de,le->l',J_,Sqi,J_),3))
```