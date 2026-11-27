The local re-implementation exists: bundle builder, encoders, prelude attention, full step and error checks. Since the edit injects through the action encoder and the prelude, I first need the encoders to be exact. Next I'll run the z-posterior check, which compares my computed posterior mean with the tool's states, and break the error down by bundle type.

```python
r=zpost_err()
print(type(r))
if isinstance(r,dict): print({k:(v if np.isscalar(v) else np.round(np.asarray(v,dtype=float)[:5],3)) for k,v in list(r.items())[:15]})
elif isinstance(r,tuple): print([np.round(np.asarray(a,dtype=float).ravel()[:6],3) for a in r])
else: print(np.round(np.asarray(r,dtype=float).ravel()[:10],3))
```