With fresh random z draws the head replica matches the tool's predicted means to MAE 1.3, which is close to the tool's own 4-draw sampling noise. Renormalising over bins 1..16 changes nothing, since p(no value) is about 0. That is accurate enough to fit a readout and gate.

Now the gating question. I will compute per-code means of `with_code` and the pooled within-code covariance over all readings and z draws. Then I will take the Mahalanobis distances between glucose and MAP codes. Distances of about 20 or more mean one linear gate can switch a unit fully on for glucose and fully off for MAP with margin. Small distances mean I must gate differently.

```python
WC=with_code(Q4,fL,lbL,cidL)            # N,4,192
ev=RT['event'].values
Xs_={c:WC[torch.tensor(ev==c)].reshape(-1,192).numpy() for c in MEAS}
mu={c:Xs_[c].mean(0) for c in MEAS}
Sw=sum(np.cov(Xs_[c].T)*(len(Xs_[c])-1) for c in MEAS)/sum(len(Xs_[c])-1 for c in MEAS)
ev_,V_=np.linalg.eigh(Sw); print('Sw eig min/med/max',ev_[:3].round(6),np.median(ev_).round(4),ev_[-1].round(3))
Si=np.linalg.inv(Sw+1e-4*np.eye(192))
for a in ['glucose_finger','glucose_lab']:
    for b_ in ['map_arterial','map_cuff']:
        d_=mu[a]-mu[b_]; print(a[:9],b_[:9],'Mahal',round(float(np.sqrt(d_@Si@d_)),1))
print('n',{c:len(Xs_[c]) for c in MEAS})
```