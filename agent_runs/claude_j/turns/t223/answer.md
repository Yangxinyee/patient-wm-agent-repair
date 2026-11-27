Still 49 mg/dL, even with my replica states passed as X. So the copied pre-restart states come from neither the X argument nor the global S I overwrote. One caveat: in this run I had reset S to the original before testing, so the run 197 test is the informative one, and it also failed. Most likely X_of was defined in a different globals namespace, or reads the stored file another way.

Two checks:
1. Is X_of's globals dict the kernel's?
2. For stay …024, how far are the stored states from the replica's at each early bundle?

If the stored states for that stay diverge strongly at bundles 1–25, the fwd_3 rows are misaligned for some stays. Then I'll make my own restart function that seeds from replica states instead of fixing X_of.

```python
print('X_of globals is kernel globals:', X_of.__globals__ is globals(), '| run_cf same:', run_cf.__globals__ is globals())
s=2020000024; Ob,Hb,_=run_cf(s,{},BUN[s][1][0],Xs); Xo=np.asarray(X_of(s))
d=np.abs(Ob.numpy()-Xo[:,:192]).max(1); print('stay024 out diff per bundle q50/max early(1-25) %.2f/%.2f late(26+) %.2f/%.2f'%(np.median(d[1:26]),d[1:26].max(),np.median(d[26:]),d[26:].max()))
s=2020000000; Ob,Hb,_=run_cf(s,{},BUN[s][1][0],Xs); Xo=np.asarray(X_of(s)); d=np.abs(Ob.numpy()-Xo[:,:192]).max(1); print('stay000 early %.2f late %.2f'%(np.median(d[1:26]),np.median(d[26:])))
```