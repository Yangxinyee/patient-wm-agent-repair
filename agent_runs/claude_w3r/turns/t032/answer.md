The tensor list gives the key structure. `attr_heads.value_bin` is a two-layer MLP (192→192→17) on `with_code`, plus a direct logit table `embed_transition.value_bin` (13 codes × 17 last-bins → 17). `last_numeric` takes 8 inputs, `embed.numeric` takes 6, and `z_post` reads [ctx, e_k, u_k] (576). The cleanest controllable injection point is the value head, which is linear in its hidden units. If the state (view) carries insulin or norepinephrine dose information, I can add a code-specific mean shift by repurposing a hidden unit in its linear regime.

Next I will read vocab.json (config, codes, normalisation, bin edges/centres). I need it to reconstruct the head-side computation from states and to see which bins correspond to which values.

```python
V = VOCAB
print(V.keys())
print('config:', json.dumps(V.get('config'), indent=None)[:3000])
print('codes:', V.get('codes'))
for k in V.keys():
    if k not in ('config','codes'):
        s=json.dumps(V[k]); print(k, len(s), s[:1500]); print('---')
```