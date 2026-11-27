Only one outcome of the last run matters.

The head's input q carries the most recent action-A rate well (R² 0.83). But most of that comes from readings the model already sees: the part of that rate not implied by the readings has R² only 0.31. The earlier action-A rate and both action-B rates are almost absent beyond the readings (R² 0.04–0.08).

A probe read inside the value head would therefore be far too noisy for action B and for lag 2. The rate signal has to be written into the state upstream, where actions enter: action encoder → prelude → b_in → h → out.

To design that offline I need a faithful reconstruction of one core step, so I'll check what tools I already have. I'll print the signatures of my earlier step and encoder functions and the names of the non-core tensors (gap, z_prior/z_post, z_in and so on). I expect step-level helper functions to exist. If none do, I'll write the step from scratch and test it against the recorded states.

```python
import inspect
for fn in ['enc_act','enc_a','set_enc','enc_variant','ctx_of','ctx_best','zpost_variants','post_err','z_act','build','build12','mk_best','predict3','head_pred','atom_vec','mlp','mlp_a']:
    f = globals().get(fn)
    try: print(fn, str(inspect.signature(f))[:90])
    except Exception as e: print(fn, 'n/a')
other = [n for n in NAMES if not n.startswith(('core.', 'encode_action.', 'attr_heads.value_bin'))]
print(len(other)); print(', '.join(other))
```