`make_edit` returns the same four tensors as ED. The earlier failure came from my own print loop: `CENT` is a dict of bin centres per code, not an array. Now I can run the comparison I intended. It checks whether ED equals make_edit at some coefficients, and whether the edit scales linearly in the coefficient.

```python
E2=make_edit(2.0,2.0)
for k in ['W0','b0','W2','TR']:
    print(k,'|E1-O| %.3f |E2-O| %.3f |ED-O| %.3f |ED-E1| %.3f'%(float((E1[k]-ORIG[k]).abs().sum()),float((E2[k]-ORIG[k]).abs().sum()),float((ED[k]-ORIG[k]).abs().sum()),float((ED[k]-E1[k]).abs().sum())))
for u in units:
    a1=(E1['W2'][:,u]-ORIG['W2'][:,u]); ae=(ED['W2'][:,u]-ORIG['W2'][:,u]); a2=(E2['W2'][:,u]-ORIG['W2'][:,u])
    r=float((ae*a1).sum()/(a1*a1).sum()); r2_=float((a2*a1).sum()/(a1*a1).sum())
    print('unit',u,'W2: ED/E1 %.3f E2/E1 %.3f resid %.4f'%(r,r2_,float((ae-r*a1).abs().max())),
          '| W0row ED-E1 %.4f'%float((ED['W0'][u]-E1['W0'][u]).abs().max()),'| b0 ED %.2f E1 %.2f O %.2f'%(float(ED['b0'][u]),float(E1['b0'][u]),float(ORIG['b0'][u])))
    print('     W2 col E1:',np.round(a1.numpy(),3))
print({k:np.round(npv(v),1) for k,v in CENT.items()})
```