The glucose/MAP separation is enormous (LDA d′ ≈ 198) only for a direction that z draws move a lot. Along it, z-noise sd is 256 versus 28.5 within-group at z = mean. A naive gate or decoder would therefore become very noisy under the tool's z sampling and in rollouts. I must include the z-draw covariance in every variance I minimise.

Next I'll:
1. Recompute the separation with z-noise included, to get the honest gate SNR.
2. Assemble the edit rows with their dose changes per lag, including the readout's "reached" flag (hours after the first edited infusion).
3. Check the linear response, i.e. how well a ridge map from Δx recovers the lag-1 dose change, for insulin and NE.

A d′ still above about 10 with z-noise would make a gate feasible.

```python
Sz = np.einsum('nki,nkj->ij', DQf.astype(np.float64), DQf.astype(np.float64))/(2*N_)
gld2=np.linalg.solve(Sw+Sz+1e-3*np.eye(192), mg_-mm_); pg2=Xf@gld2; zn2=(DQf[:,0,:]@gld2).std()
sdw=np.sqrt(0.5*(pg2[isg].var()+pg2[ism].var())); print('LDA2 dprime w/ z', ((pg2[isg].mean()-pg2[ism].mean())/np.sqrt(sdw**2+zn2**2)).round(2), 'sdw', sdw.round(3), 'zsd', zn2.round(3))
EDR={}
for key in [('insulin',2.0),('insulin',0.5),('norepinephrine',2.0),('norepinephrine',0.5)]:
    gi_,dX_,dd_,rc_=[],[],[],[]
    for ii_,g_ in enumerate(GEN):
        if key not in g_['ED']: continue
        idx_=np.where(g_['sel'])[0]; hh_=g_['hour'][idx_].astype(int); dose_=g_['ins'] if key[0]=='insulin' else g_['ne']
        fdh=[j_-1 for j_ in range(13,27) if dose_[j_]>0]; first=min(fdh) if fdh else 99
        L_=np.zeros((len(idx_),4))
        for k_ in range(1,5):
            jj=hh_-k_+1; ok_=(jj>=13)&(jj<27); L_[ok_,k_-1]=(key[1]-1)*dose_[jj[ok_]]
        gi_.append(offs[ii_]+idx_); dX_.append(g_['ED'][key]); dd_.append(L_); rc_.append(hh_>first)
    EDR[key]=dict(gi=np.concatenate(gi_), dX=np.concatenate(dX_).astype(np.float64), dd=np.concatenate(dd_), reached=np.concatenate(rc_))
    e_=EDR[key]; oc=(cf[e_['gi']]==(0 if key[0]=='insulin' else 2))&e_['reached']
    from sklearn.linear_model import Ridge
    rr_=Ridge(alpha=1.0).fit(e_['dX'][oc], e_['dd'][oc,0]); 
    print(key, 'rows', oc.sum(), 'mean dd1', e_['dd'][oc,0].mean().round(3), 'R2 dd1~dX', round(rr_.score(e_['dX'][oc], e_['dd'][oc,0]),3), '|dX| mean', np.linalg.norm(e_['dX'][oc],axis=1).mean().round(3))
```