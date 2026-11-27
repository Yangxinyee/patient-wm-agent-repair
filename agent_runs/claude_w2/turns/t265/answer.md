`W2` changed only one tensor, `core.prelude.0.attn.out.weight`, with entries up to 925. The earlier attempt took a decoder (`wI48`, norm 223) on head 2's existing 48-dim value for u and amplified it by about 650 into a slot direction (`vI`). That amplifies the existing curved, bin-like dose encoding, which run 252 showed cannot respond linearly to raised doses. No wonder I moved away from it.

The new plan builds the channel from the one linear dose input, the numeric z_rate column of `embed.numeric`. It goes through gated, linear-regime units in `encode_action.mlp1` and `mlp2` into u, then through head 2 (attention ≈1) into the slots. To build it I need the exact encoder code path: how atoms are embedded and how mlp1, the sum, the size term and mlp2/norm combine. I'll print the signatures of my encoder functions and the impl.py lines that implement the set encoder.

```python
for f in ['make_atoms','encode_bundles','embed_atoms','atom_feats','bundle_enc','encode_from_xa','set_encode','embed','atom_features','empty_enc','encode']:
    fo=globals().get(f)
    try: print(f, str(inspect.signature(fo))[:120] if callable(fo) else 'MISSING')
    except Exception as ex: print(f,'?')
src=open('/work/impl.py').read().split('\n')
for i,l in enumerate(src):
    if any(k in l for k in ['mlp1','mlp2','sqrt','size','def encode','def embed','numeric','def set_']): print(i, l[:150])
```