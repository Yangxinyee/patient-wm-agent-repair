For stay …024 the stored `out` vectors match the replica's to 0.01–0.02 at every bundle. So the mismatch must be in the stored h slots: my earlier "H vs stored" check compared run_cf's copy with itself. The stored slots may use a different layout (for example 192×4 instead of 4×192) or a different stage. The heads read the slot mean after the flow decay, and the bad early-hour predictions are exactly where stored slots are used.

Next I compare the replica's H with the stored H per bundle under both reshape conventions.
- If the transposed layout matches, the stored slots were just mis-reshaped.
- If neither matches, the stored h is a different stage, and I'll recompute from bundle 1 everywhere.

```python
s=2020000024; Ob,Hb,_=run_cf(s,{},BUN[s][1][0],Xs); Xo=np.asarray(X_of(s)); Hr=Hb.numpy()
A=Xo[:,192:960].reshape(-1,4,192); B=Xo[:,192:960].reshape(-1,192,4).transpose(0,2,1)
for nm,Z in (('4x192',A),('192x4T',B)):
    d=np.abs(Hr-Z).max((1,2)); print(nm,'early q50 %.3f max %.3f | late q50 %.3f'%(np.median(d[1:26]),d[1:26].max(),np.median(d[26:])))
dm=np.abs(Hr.mean(1)-A.mean(1)).max(1); print('slot-mean diff early %.3f late %.3f'%(np.median(dm[1:26]),np.median(dm[26:])))
```