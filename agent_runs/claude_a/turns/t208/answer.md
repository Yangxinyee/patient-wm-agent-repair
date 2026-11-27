The free code gate works. A linear discriminant on the value-MLP input separates glucose_finger from all other codes with held-out scores of at least 0.983 against at most −0.986. map_arterial separates just as cleanly.

So the readout can be gated with a large multiplier on this discriminant, with no extra killed dim and no atom-encoder cut. I will refit it on the killed scaffold, excluding the circuit dims, so the memory signal cannot leak into the gate.

This suggests an even leaner write side: drop the staging dims, and let the two gated GELU units in block 0.0 read the insulin rate directly from that FFN's normalized input (`n3_0`). That needs only three killed dims: a reference jR and memories jI and jN.

To fit the gate and the rate decoder, I need the per-bundle internals:
- hf and E from `tf_loop`;
- the cross-attention memory from `mem_for`;
- `n3_0` from `core_rest(..., keep=True)`.

This run collects them on dsE and checks that the recomputed hn matches the `tf_loop` store. If it matches to about 1e-3, the internals pipeline is exact and I can fit on it. If not, I need to find which input is wrong.

```python
t0=time.time() if 'time' in globals() else None
import time; t0=time.time()
Hs,Os,ST=tf_loop(PO, dsE, keep=('E','hf','hn'))
print('store', {k:tuple(v.shape) for k,v in ST.items() if hasattr(v,'shape')}, round(time.time()-t0,1),'s')
Mx,Ma,Mm=mem_for(dsE, dsE['srows'], torch.as_tensor(dsE['e']))
idx=torch.arange(0,2000)
hn_,y_,out_,inf_=core_rest(PO, ST['hf'][idx], ST['E'][idx], Mx[idx], Ma[idx], Mm[idx], keep=True)
print('hn err', float((hn_-ST['hn'][idx]).abs().max()), 'n3_0', tuple(inf_['n3_0'].shape), round(time.time()-t0,1),'s')
```