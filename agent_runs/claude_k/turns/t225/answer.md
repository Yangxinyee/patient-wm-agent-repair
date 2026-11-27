Run 208 gives the alignment I need. Each reading row in `R` carries:
- `k`, the state row the model used to predict it, so the reading's own bundle is row `k+1` of `S2`;
- the tool's teacher-forced predicted mean, `mean_rec`.

`S2.x` holds per bundle `out` (192), the 4 slots (768) and the z posterior mean (32). So for every glucose and MAP reading I can pair the surprise (value minus `mean_rec`) with the z posterior of its bundle.

To use the surprise part of z, I need the z prior at that bundle. That means rebuilding `ctx` (`out` + `out_norm` of decayed slots + `gap`) from the previous state. My kernel has helper functions for this, but I don't remember their signatures. I'll list the signatures of the relevant helpers and the gap-module tensor names. That should let me compute the prior next run without rewriting code.

```python
import inspect
fns=['ctx_of','u_rec','u_c','datom','set_enc','enc_act','z_act','head_pred','head_means','probs_from','gelu','mlp','lin','zsc','fbin','atom_vec','enc_a','mlp_a','L64','base_mk','mk_best','ctx_best']
for f in fns:
    if f in globals():
        try: print(f, str(inspect.signature(globals()[f]))[:120])
        except Exception as e: print(f,'?',e)
print([n for n in NAMES if 'gap' in n or 'filter' in n][:12])
print([k for k in globals() if 'view' in k.lower() or 'prior' in k.lower() or 'setenc' in k.lower()])
```