`kill(P0, J, mem)` already zeroes all readers and writers of chosen state dimensions in the prelude, blocks, `c_out`, coda, out_norm, gap, z_head, type_emb and query_norm. It does not zero the prelude attention pre-norm, which matters only for dimensions that hold memory. So the scaffold for a memory pathway is ready.

Plan for the insulin pathway:
- **Write:** two GELU units in the prelude FFN gate a linear rate readout so it is on only at drug bundles. They write into a cut staging dimension, which `b_in` adds into a cut memory dimension (abar=1, time decay via `flow.a`).
- **Read:** one glucose_finger-gated value-MLP unit applies a logit tilt to that memory, minus a regression-based compensation.

Before fitting anything, I need to recall which data frames and helper functions exist in the kernel. I'll list user functions, the train-frame columns and a few rows, keeping the output short.

```python
import types
fn=sorted(k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_'))
print(len(fn),' '.join(fn)[:1500])
print(list(tr.columns)); print(tr.head(4).to_string()[:700])
dv=sorted(k for k,v in globals().items() if isinstance(v,dict) and not k.startswith('_'))
print('dicts:',' '.join(dv)[:600])
```