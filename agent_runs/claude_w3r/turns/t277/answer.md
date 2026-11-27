Still bad: the residual is about 0.5 and the grid is distorted, even with a weak constraint. The likely cause is my ridge term. With a per-sample-normalised Gram, `lam=1e-3·I` probably swamps the informative directions. The kernel's `ridge_read` also returns a 193-vector (it has an intercept) plus four scalars.

I will print those scalars, which are probably the residual and leakage, and retry my read with much weaker ridge values. If the residual drops to about 0.1 with a linear grid, the ridge was the cause.

```python
print('ridge_read scalars', [round(float(t),4) for t in r0[1:]])
for lam in [1e-5,1e-7]:
    for lamn in [1,10]:
        for j,nm,gr in [(0,'ins',gi),(1,'ne',gn)]:
            w=rr2(pa[:,j],lamn,lam); lk=NEG@w
            rg=[float(n1_of(u_of(**{nm:r}))@w) for r in gr]
            print(lam,lamn,nm,'res',round(float(np.std(U_act@w-pa[:,j])),3),'leak sd',round(float(lk.std()),3),'max',round(float(np.abs(lk).max()),2),'u0',round(float(b1n@w),3),'|w|',round(float(np.linalg.norm(w)),1),'grid',np.round(rg,2))
```