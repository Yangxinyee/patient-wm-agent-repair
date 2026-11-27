The normalisation is linear in the rate: z = (rate − median) / IQR, with median 1.5 and IQR 1.0 for action A, and median 0.46 and IQR 0.59 for action B. Dose z mirrors rate z, scaled by a fixed factor. So the encoder's numeric inputs are an exactly linear carrier of the rate, except for clipping at z = 10. That only bites for action-A rates above 11.5, i.e. some ×3 edits.

The state, including the posterior z, loses most of the rate information beyond the readings. To see where, I need u_k. My old helpers `atom_vec`, `set_enc` and `enc_a` should compute it, but I've lost their calling conventions, so I'll probe them with trial calls. If they work, I'll compute u_k for the 2,491 action bundles and test its R² for the rates. High R² would mean the loss happens downstream, in z_post or the prelude.

```python
tests = {}
try:
    v = atom_vec(8, kind=0, ekind=2, zd=0.5, hd=1.0, zr=0.5, hr=1.0, dbin=9, rbin=9); tests['atom_vec'] = (np.shape(v), float(np.linalg.norm(v)))
except Exception as e: tests['atom_vec'] = repr(e)[:120]
for p in ['encode_action', 'encode_action.', 'act']:
    try:
        o = set_enc(np.stack([v, v]), p); tests['set_enc ' + p] = (np.shape(o), float(np.std(o)))
    except Exception as e: tests['set_enc ' + p] = repr(e)[:120]
try:
    o = enc_a(np.stack([v, v]), 'encode_action', gelu_np, 'log'); tests['enc_a'] = (np.shape(o), float(np.std(o)))
except Exception as e: tests['enc_a'] = repr(e)[:120]
for k, r in tests.items(): print(k, '->', r)
print([n for n in vars_ if 'unit' in n.lower() or 'kind' in n.lower() or 'EK' in n][:20])
```