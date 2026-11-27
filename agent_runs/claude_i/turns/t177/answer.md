Decodability from the state after a drug bundle is clear-cut. The raw insulin rate is well decodable (CV R² 0.85 at lag 0, falling to 0.67 at lag 3), but mostly through glucose, which drives the dose. The deviation from the policy's expected rate (rate − pibar), which is the only exogenous variation, is barely represented: R² 0.24 at lag 0, 0.23, 0.14 and about 0 by lag 3. NE rates sit around R² 0.63.

So a head-only readout of the dose deviation from the current state cannot work. The state has to be made to carry the deviation, which means editing the update at drug bundles. The natural site is z_post, the only place that sees both ctx (which knows the glucose just read) and u (the dose).

To build "rate − expected rate" there, I need two linear readouts:
1. Rate from u, fitted so that bundles without insulin give 0. The NE-only and empty-set u vectors make this non-trivial.
2. pibar(g) from ctx or from the reading bundle's e.

I need the correct current encoders for this; my notes say empty sets are encoded rather than zeroed, which enc.py's `bundle_eu` gets wrong. Since source is not retrievable, I'll check the signatures of the candidate functions and see which ones reproduce the tool's u.

```python
import inspect
for f_n in ['eu_routed','eu_routed2','eu_with_end','u_from_V','u_only','u_only_x','encode_set','set_enc','atom_vec','atom_end','atoms_embed','core_step','pred_mean_s','head_value2','zprior','view_ctx','true_shift','drug_after','run_var','encode_bundles','build_bundles']:
    try: print(f_n, inspect.signature(globals()[f_n]))
    except Exception as ex: print(f_n, 'ERR', ex)
```