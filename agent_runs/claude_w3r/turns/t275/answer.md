The lowest-energy value directions have energy of about 1e-4 (sd ≈0.012), not 1e-9. Repurposing them would remove only a tiny original contribution: at most |Wo2·n| ≈ 6 × 0.012, against slot tokens of norm about 800. So they are usable as nearly free channels.

The run then crashed on a float `.round` call. I am rerunning the read fit with that fixed, to check that the rate reads are clean and roughly linear.

```python
for lam in [1e-4,1e-3]:
    w_i=rr(pa[:,0],lam=lam); w_n=rr(pa[:,1],lam=lam)
    gi=[.25,.5,1,1.5,2,3,4,5,6,8,10]; gn=[.03,.1,.25,.5,.75,1,1.25,1.5,2]
    ri=[float(n1_of(u_of(ins=r))@w_i) for r in gi]; rn=[float(n1_of(u_of(ne=r))@w_n) for r in gn]
    print(lam,'ins res',round(float(np.std(U_act@w_i-pa[:,0])),3),'NEGmax',round(float(np.abs(NEG@w_i).max()),3),'u0',round(float(b1n@w_i),3),'grid',np.round(ri,2))
    print(lam,'ne  res',round(float(np.std(U_act@w_n-pa[:,1])),3),'NEGmax',round(float(np.abs(NEG@w_n).max()),3),'u0',round(float(b1n@w_n),3),'grid',np.round(rn,2))
    print(' cross: ins-read on ne-only', np.round([float(n1_of(u_of(ne=r))@w_i) for r in [.5,1,2]],2), 'ne-read on ins-only', np.round([float(n1_of(u_of(ins=r))@w_n) for r in [1,3,6]],2))
```