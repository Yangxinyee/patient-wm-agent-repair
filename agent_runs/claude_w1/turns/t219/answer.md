Insulin x2 response 0.26 against a dose change of 1.6, noise about 0.69. That is too weak.

Next: the best achievable signal-to-noise for any linear decoder, sqrt(g'Σ⁻¹g). Here g is the mean head-input change per unit lag-1 dose change in the own-drug x2 edit, and Σ is the factual covariance.

```python
def best_snr(key,codes,rw,lam=1e-3):
    e_=EDR[key]; m_=np.isin(cf[e_['gi']],codes)&e_['reached'].astype(bool)
    g_=e_['dX'][m_].mean(0)/e_['dd'][m_][:,0].mean()
    S_=np.cov(Rf[rw].T.astype(np.float64)); S_+=lam*np.trace(S_)/192*np.eye(192)
    w0=np.linalg.solve(S_,g_); w0/= (w0@g_)
    return w0, np.sqrt(g_@np.linalg.solve(S_,g_))
for key,codes,rw in [(('insulin',2.0),[0,1],ins_rows),(('norepinephrine',2.0),[2,3],map_rows)]:
    for lam in [1e-4,1e-3,1e-2,1e-1]:
        w0,snr=best_snr(key,codes,rw,lam)
        print(key[0][:3],lam,'snr',round(snr,3),'noise/unit',round(np.std(Rf[rw]@w0),3),[(k[0][:3]+str(k[1]),resp(w0,k,codes)[0]) for k in keysI+keysN])
```