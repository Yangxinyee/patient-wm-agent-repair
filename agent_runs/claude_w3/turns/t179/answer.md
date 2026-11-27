h is large: about 65 per-bundle sd across dims, roughly 11% variation, with per-dim changes around 8.6. `out` is a LayerNorm output whose per-dim variation is small (about 0.1). A dedicated memory with values of order 10–50 in one h dimension would barely disturb LayerNorm statistics, so the design is feasible.

Next I'll check the signatures of the local re-implementation functions already in the kernel (encoders, prelude, step) so I can reuse them rather than rewrite.

```python
import inspect
for fn in ['build_bundles','stay_bundles','enc_bundle','atom_vec','set_enc_v','bundle_vecs','prelude_attn','mha','step','step_err3','zpost_err','zparts','LN','lin','mlp']:
    f=globals().get(fn)
    try: print(fn, inspect.signature(f))
    except Exception as e: print(fn,'?',e)
```