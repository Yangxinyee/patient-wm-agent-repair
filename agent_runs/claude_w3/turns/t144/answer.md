Both block layers have self-attention, cross-attention to the memory, and an FFN, each with its own norm. The saved state file holds 644 bundles with 992 numbers each (out, the 4 h slots, the z posterior mean), plus stay, hour and drug flags.

The cleanest test of my encoders is to rebuild the z posterior mean of bundle k from the saved state of bundle k-1 plus my encoded e_k and u_k. That isolates the encoders and the context term from the harder recurrent block. First I need the signatures of the encoder helpers I already have.

```python
for fn in ['stay_bundles','enc_bundle','bundle_vecs','atom_vec','atom_emb','set_enc','set_enc_v','mlp','mlp_','lin','lin_','ln_','LN','mha','HN','zparts','zpost_err','G_','A']:
    f = globals().get(fn)
    if f is not None and inspect.isfunction(f): print(fn, inspect.signature(f))
```