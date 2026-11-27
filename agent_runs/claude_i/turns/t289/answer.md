The signatures are what I hoped for. `step_local(k, ..., ztok_add=...)` runs one model step with an addition to the z token. `value_mean` and `pred_mean_s` reproduce the heads. The `states` file contains stay ids, hours and drug flags.

The run failed only because `load` returns torch tensors, which have no `.astype`. I'll convert them to float64 numpy and repeat the checks:
- z_post reproduces z (expected error about 1e-6);
- the dead hidden units and their pre-activations;
- how variable z dimensions 18 and 28 are.

I'll also call `step_local` once to learn what it returns, since the perturbation analysis will rely on it.

```python
N = lambda t: np.asarray(t.numpy() if hasattr(t,'numpy') else t, dtype=np.float64)
W0=N(load('z_post.0.weight')); b0=N(load('z_post.0.bias')); W2=N(load('z_post.2.weight')); b2=N(load('z_post.2.bias')); Wzi=N(load('z_in.weight')); bzi=N(load('z_in.bias'))
Xin = np.concatenate([N(C2),N(Eall),N(Uall)],1)
pre = Xin@W0.T + b0
hid = gel(pre); zm = hid@W2.T + b2
X3n = N(X3)
print('z err', np.abs(zm[:,:32]-X3n[:,960:]).max(), np.median(np.abs(zm[:,:32]-X3n[:,960:])))
sd = hid.std(0); dead = np.where(sd<0.01)[0]; print('ndead', len(dead), 'pre max over dead', pre[:,dead].max().round(2), 'mean', pre[:,dead].mean().round(2))
print('z18/z28 sd', zm[:,18].std().round(4), zm[:,28].std().round(4), 'mean', zm[:,18].mean().round(3), zm[:,28].mean().round(3))
r = step_local(100)
print(type(r), [ (type(x), getattr(x,'shape',None)) for x in (r if isinstance(r,(tuple,list)) else [r])])
sid3 = zf['stay_id']; hr3 = zf['hour']; hI = zf['has_insulin']; hN = zf['has_norepinephrine']
print('bundles', len(sid3), 'ins', hI.sum(), 'ne', hN.sum(), 'stays', len(np.unique(sid3)), 'hour range', hr3.min(), hr3.max())
```