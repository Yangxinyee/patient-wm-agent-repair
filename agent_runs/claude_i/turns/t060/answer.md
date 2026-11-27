The head-only route is out. The state encodes the glucose reading almost perfectly (R² 0.97), but the dose's deviation from what the policy would give hardly at all:

| Decoding from | D_t R² | dev_t R² | dev_t−1 R² |
|---|---|---|---|
| Head input q | 0.72 | 0.21 | 0.19 |
| Hidden layer | 0.59 | −0.02 | 0.10 |
| Full 960-dim state | 0.69 | 0.19 | 0.16 |

The D_t R² mostly reflects the reading, since dose ≈ policy(reading). So the core loses most of the dose-magnitude information before the heads can see it. That is why the model's dose response is about 5× too weak.

Any working edit must therefore make the action pathway carry dose magnitude into the state. That means an edit in the action encoder, the z posterior or the prelude, which I can only design with a replica of those pieces.

I'll first see what my earlier replica functions (`atom_vec`, `encode_set`, `zpost_cand`) take and do, by inspecting their argument names, defaults, constants and referenced globals, instead of rewriting from scratch. If they are usable, I'll validate them on drug-only bundles, where z_post([ctx, 0, u_k]) must equal the stored posterior mean. Otherwise I'll rebuild the encoder myself.

```python
for fnm in ['rn','lin','ln','mlp','encode_set','atom_vec','zpost_cand']:
    f_ = globals()[fnm]; co = f_.__code__
    print("==", fnm, "args:", co.co_varnames[:co.co_argcount], "defaults:", f_.__defaults__)
    print("   names:", co.co_names[:40])
    print("   consts:", [c for c in co.co_consts if not hasattr(c,'co_code')][:40])
```