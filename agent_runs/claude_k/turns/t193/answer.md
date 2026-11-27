The call itself works; my wrapper was the problem. `view_from` returns a **3-tuple** of arrays with different shapes, and `np.asarray` on that tuple raised the "inhomogeneous shape" error. Earlier I hadn't wrapped the call, so it ran.

I'll look at the three returned elements, find which one is q by comparing with `Q99[0]`, and then compute q for both models on all readings (batched if the function supports it).

```python
r_=view_from(X8o[kidx[0]][None,:], np.array([3600.0]))
print(type(r_), len(r_), [np.asarray(x).shape for x in r_])
for j__,x in enumerate(r_):
    x=np.asarray(x)
    if x.shape[-1]==192: print(j__,'err vs Q99[0]',np.round(np.abs(x.ravel()-Q99[0]).max(),6))
rB=view_from(X8o[kidx], dts99); print('batched shapes',[np.asarray(x).shape for x in rB])
```