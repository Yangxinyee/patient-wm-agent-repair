No combination works: the error stays around 0.4 whatever the gap or flow settings. The flow hardly matters at all. So the remaining problem is elsewhere: the activation inside `z_post`, the order of its inputs, or how ctx is formed.

Two things I can check from leftovers:
- the `ACTMAP` I set earlier, since the encoders match with whatever activations it assigns;
- whether `recs` also holds the probe's ctx projections, from the 10 rows of P that read ctx.

If it does, I can test ctx directly against forward pass 3's states, separately from `z_post`.

```python
print('ACTMAP', ACTMAP)
print(len(recs[0]), [ (type(x).__name__, getattr(x,'shape',x) if not isinstance(x,(int,np.integer)) else x) for x in recs[0][:8]])
npz3=np.load('/work/fwd_3/states_edited.npz'); print(npz3['x'].shape)
pe=np.load('/work/probe_zpost_edit.npz'); print({k:pe[k].shape for k in pe.files})
W1p=pe['z_post_0_weight']; b1p=pe['z_post_0_bias']; W2p=pe['z_post_2_weight']; b2p=pe['z_post_2_bias']
print('b1 uniq', np.unique(np.round(b1p,2))[:6], 'nnz rows W1', int((np.abs(W1p).sum(1)>0).sum()), 'b2[:32] uniq', np.unique(np.round(b2p[:32],2))[:6])
```