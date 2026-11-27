The tensor list fills in most of the architecture. The heads are 2-layer MLPs (`.0` → activation → `.2`); there are transformer layers with separate q and kv projections, norms and FFNs; `z_post` takes 576 = 3 × 192 inputs (ctx, e, u). Some details are still unknown: activation functions, attention scaling, the norm type, the exact `size`/`gap`/`age` MLPs, and how the time-flow decay interacts with the slots.

Next I'll check whether tensors.json carries per-tensor descriptions beyond shapes. If it explains each tensor's role, that cuts the guesswork in a local re-implementation.

```python
for k in list(TJ.keys())[:12]: print(k, json.dumps(TJ[k])[:400])
print(len(TJ))
others = [k for k in TJ if k not in NAMES]; print('keys not in NAMES:', others[:20])
```