The shape list confirms what the design needs:

- **Per-dimension state controls.** The time-flow decay `core.flow.a` and the update gates `core.a_in` and `core.delta_in` are separate for each of the 192 h dimensions. So a few h dimensions could, in principle, be given their own persistence and act as a dose memory.
- **Bias-free attention.** All attention projections (`q`, `kv`, `out`) lack biases. `b_in` and `c_out` are plain 192×192 matrices.
- **Value head and transition table.** The value head is a 192-unit GELU MLP over 17 bins. `embed_transition.value_bin` is a 221×17 table indexed by code × last bin, which allows exact per-code logit offsets.

**Main risk: the write path.** The only way a dose reaches h is through the prelude's attention over the 7 tokens (4 slots, e, u, z_in). If the slots' attention to the u token (or to the z_in token) varies a lot from bundle to bundle, any dose memory written this way will have an inconsistent gain.

**Next step.** I will check the data structures still in the kernel:

- the atoms table AT and the bundle table SB, with their columns;
- whether the per-bundle encodings e and u are still stored;
- what `prep_stay` returns.

With these I can compute the prelude attention weights directly for real action bundles.

**What would confirm feasibility:** one head whose slot-to-u attention stays high and stable on action bundles. **What would refute it:** attention to u that is weak or very variable. In that case I would route the write through a specially built key/value direction instead.

```python
import numpy as np, torch
cut = [k for k in list(globals().keys()) if not k.startswith('_')]
i0 = cut.index('rows_x') if 'rows_x' in cut else 0
rest = [k for k in cut[i0:] if not callable(globals()[k])]
print('REST VARS:', ' '.join(f"{k}:{type(globals()[k]).__name__}{tuple(globals()[k].shape) if hasattr(globals()[k],'shape') else ''}" for k in rest))
print(AT.columns.tolist()); print(AT.head(8).to_string())
print(SB.columns.tolist()); print(SB.head(5).to_string())
print(B.columns.tolist(), MB.columns.tolist())
try:
    pr = prep_stay(tu, sel1[0])
    print('prep_stay ->', type(pr), (list(pr.keys()) if isinstance(pr, dict) else len(pr)))
    if isinstance(pr, dict):
        for k_,v_ in pr.items(): print('  ', k_, type(v_).__name__, getattr(v_,'shape', v_ if np.isscalar(v_) else (len(v_) if hasattr(v_,'__len__') else '')))
    else:
        for v_ in pr: print('  ', type(v_).__name__, getattr(v_,'shape', ''))
except Exception as ex:
    print('prep_stay error', repr(ex))
```