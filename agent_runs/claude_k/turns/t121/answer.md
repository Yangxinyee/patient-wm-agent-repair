`atom_vec` works and returns a 192-dim atom embedding. `set_enc` and `enc_a` fail with "NoneType is not callable", so they rely on a global activation that has since been overwritten (I reused the name `act` for a table). Rather than chase my old code, I'll rebuild the encoder cleanly from the tensors. Its form is fixed by the architecture: LayerNorm(mlp2(Σ mlp1(atom)/√n + size(log(1+n)))), with GELU and eps 1e-5.

For that I need the categorical conventions for drug atoms: event and atom kinds, the categorical table sizes, and what the `pending` vector p is. I'll print those compactly, plus the type of `act_`.

```python
print('event_kinds', VOCAB['event_kinds']); print('atom_kinds', VOCAB['atom_kinds'])
for n in NAMES:
    if n.startswith(('embed.categorical', 'embed.kind', 'embed.ekind', 'embed.kq', 'embed.text', 'pending_mlp.0', 'filter_gate', 'last_status', 'type_emb', 'gap.0', 'z_in', 'z_post.0', 'z_post.2', 'z_prior.0')):
        print(n, tuple(SHAPES[n]), end='; ')
print(); print('act_ type', type(globals().get('act_')).__name__)
```