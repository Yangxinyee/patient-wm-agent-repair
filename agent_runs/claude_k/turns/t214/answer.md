The spectra change the picture. out and q are effectively low-rank: their median eigenvalues are about 1e-5, against largest eigenvalues of 25 and 4, and many directions are essentially zero. So most directions of q are naturally quiet.

My earlier SNR used a ridge of 1e-4 to 1e-3 on the eigenvalues. That floored exactly these quiet directions and made the z channel look hopeless when it may not be. If the fwd8 writes leak even a small, consistent component into q's quiet subspace, a readout there could be nearly noise-free.

The honest test is direct decoding with stay-grouped cross-validation: predict the lagged rates from the edited q (fwd8) and compare with the original q. That captures every noise source, including the nonlinear, not-explained-by-dose part of Δq. I scan small ridge penalties.
- If edited-q R² for the previous norepinephrine rate and insulin dose rises far above original-q R² (toward 0.9 or more), the channel is viable. The design then becomes: write innovations, and read them along quiet directions.
- If R² barely moves, the leak into quiet directions is too noisy, and I will need the step rebuild.

```python
from sklearn.linear_model import Ridge
gkR=GroupKFold(5)
def cvq(X,y,grp,al):
    p=np.zeros_like(y)
    for a,b in gkR.split(X,y,grp): p[b]=Ridge(alpha=al,solver='svd').fit(X[a],y[a]).predict(X[b])
    return 1-((y-p)**2).mean()/y.var()
mG=np.isin(evR,['glucose_finger','glucose_lab']); mM=np.isin(evR,['map_arterial','map_cuff'])
for nm,msk,y in [('NE lag1 @map',mM,lagd(ne_all,1)[mM]),('NE lag2 @map',mM,lagd(ne_all,2)[mM]),
                 ('INS lag1 @gluc',mG,lagd(ins_all,1)[mG]),('INS lag2 @gluc',mG,lagd(ins_all,2)[mG])]:
    g_=sR[msk]
    res_=[]
    for al in [1e-9,1e-7,1e-5,1e-3]:
        res_.append('a%g: orig %.3f edit %.3f dq %.3f'%(al,cvq(Qo8[msk],y,g_,al),cvq(Qe8[msk],y,g_,al),cvq(dQ8[msk],y,g_,al)))
    print(nm,' | '.join(res_))
# where does the signal live? project dq response onto eigenbasis of natural q
Sig=np.cov(Qo8[mM].T); ev_,V_=np.linalg.eigh(Sig)
X_=XL[mM]; Bq_,*_=np.linalg.lstsq(X_,dQ8[mM],rcond=None); rN=Bq_[1]
proj=(V_.T@rN)**2
for lo,hi in [(0,1e-9),(1e-9,1e-7),(1e-7,1e-5),(1e-5,1e-3),(1e-3,10)]:
    sel=(ev_>=lo)&(ev_<hi); print('eig in [%g,%g): n %d, frac of |rNE1|^2 %.3f'%(lo,hi,sel.sum(),proj[sel].sum()/proj.sum()))
```