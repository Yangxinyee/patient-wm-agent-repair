The states files hold 992 numbers per bundle: out, 4 h slots, then z. I'll take slot 0 of h, layer-normalise it per bundle (no affine), and check its PCA spectrum. The question is how many dimensions hold 99%, 99.9% and 99.99% of the variance, both around its mean and uncentred.

```python
for f_ in ['/work/fwd_3/states_original.npz','/work/fwd_5/states_original.npz']:
    z_=np.load(f_, allow_pickle=True); X_=z_['x']; hs=X_[:,192:384].astype(np.float64)
    nh=(hs-hs.mean(1,keepdims=True))/hs.std(1,keepdims=True)
    for lab_,M_ in [('centred',np.cov(nh.T)),('uncentred',nh.T@nh/len(nh))]:
        ev_=np.sort(np.linalg.eigvalsh(M_))[::-1]; c_=np.cumsum(ev_)/ev_.sum()
        print(f_.split('/')[-2], lab_, len(nh), [int(np.searchsorted(c_,t)+1) for t in [0.9,0.99,0.999,0.9999]], 'top', np.round(ev_[:3],2))
```