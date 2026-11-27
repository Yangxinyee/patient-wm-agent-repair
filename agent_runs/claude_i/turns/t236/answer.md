The relative error stays at about 1.03 whatever the atom offset, so the inputs are not the problem. My target or input array must be wrong: X3 may be a processed or standardised copy rather than the raw states. I will load the raw file /work/fwd_3/states_original.npz, print its keys and shapes, and check whether it matches X3 and the z array. If the raw file differs from X3, I will rebuild from the raw file and recheck. If they match, the bug is in how I build ctx.

```python
S=np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
print({k:(S[k].shape,S[k].dtype) for k in S.files})
for k in S.files:
    a=S[k]
    if a.ndim==2 and a.shape[1]==992:
        R=a; print('raw vs X3 maxabs',float(np.abs(R-np.asarray(X3)).max()), ' raw z vs zm3', float(np.abs(R[:,960:]-np.asarray(zm3)).max()), ' raw z vs ZZt', float(np.abs(R[:,960:]-np.asarray(ZZt)).max()))
        print('row0 out mean/sd',round(float(R[1,:192].mean()),3),round(float(R[1,:192].std()),3),' z sd',round(float(R[:,960:].std()),3))
```