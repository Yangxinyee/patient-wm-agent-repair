The kernel already has much of the machinery from earlier sessions. `stay_bundles`, `atom_vec`, `set_encode`, `split_atoms` and `zpost_check`/`zpost_check2` mean I built the atom embedding and both set encoders, and checked them against the tool's z posterior. `ctx_from_state`, `z_draws` and `head_eval` cover the view and value head. So rebuilding the core needs only the prelude, the two recurrent layers with memory cross-attention, and the coda. I will print the source of the encoder and checking functions to see how they work and how well they matched.

```python
for fn in ['stay_bundles','split_atoms','atom_vec','set_encode','zpost_check2','ctx_from_state','act_fn','mlp']:
    try:
        s_=inspect.getsource(globals()[fn]); print(s_[:1800]); print('-----')
    except Exception as ex: print(fn,'ERR',ex)
```