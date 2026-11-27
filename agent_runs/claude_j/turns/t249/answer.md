Run 221 settles the head approach. The best linear readout of the head features has an out-of-fold signal-to-noise ratio of only 1.4, and just 1.6 even in-sample. The threshold was about 4. So the factual head features barely tell a doubled-dose state from ordinary factual variation, and more data cannot fix that. A head-only edit could recover perhaps a third of the missing effect before using up the factual budget.

To do better, the dose signal has to reach the state through the action encoding u_k. Next I will check how the replica represents a bundle and what `vecs` returns. Then I can test whether injecting a direction into u_k at insulin bundles yields a strong, clean glucose response at the heads. This step only gathers information. I need the tuple layout, the shapes, and a way to identify insulin bundles.

```python
b0=BUN[s0]; print(type(b0).__name__, len(b0))
k_ins=[k for k in range(len(b0)) if 'insulin' in str(b0[k])[:2000]][:3]; print('ins bundles',k_ins)
bk=b0[k_ins[0]]; print(type(bk).__name__, len(bk) if hasattr(bk,'__len__') else '', [type(x).__name__ for x in bk][:6], str(bk)[:300])
vv=vecs(s0,k_ins[0]); print(type(vv).__name__, [tuple(x.shape) if hasattr(x,'shape') else x for x in (vv if isinstance(vv,(tuple,list)) else [vv])])
```