The action encoder is a small DeepSets module: per-atom mlp1 (192→768→192), a sum over atoms plus a size term, mlp2 (192→768→192), then a LayerNorm. I can amplify the drug-rate-induced part of mlp2's output with a low-rank change to `encode_action.mlp2.2`. Its only consumers are the prelude token and z_post, so the model's own downstream machinery would carry the larger dose signal.

First I need a reimplementation of the encoder that exposes its internals and matches the helpers already verified against the tool (u_rec matched the tool-implied projections earlier). I also need the realistic rate ranges, read from the vocab normalisation and bin edges.

If my reimplementation reproduces `u_rec` to rounding error, I can build the amplification offline in the next run. If it does not, I first need to work out how the bundle's atoms are assembled.

```python
import numpy as np
T = lambda x: x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
print('codes', VOCAB['codes'])
num = VOCAB['numeric']; print('numeric type', type(num), list(num.keys())[:20] if isinstance(num, dict) else len(num))
print('note', str(VOCAB['numeric_note'])[:400])
for c in ['insulin', 'norepinephrine']:
    if isinstance(num, dict) and c in num: print(c, str(num[c])[:700])
WA = {n: T(load_original(n)).astype(np.float64) for n in NAMES if n.startswith('encode_action')}
def enc_int(V):
    V = np.atleast_2d(np.asarray(V, dtype=np.float64)); n = V.shape[0]
    h1 = gelu(V @ WA['encode_action.mlp1.0.weight'].T + WA['encode_action.mlp1.0.bias']) @ WA['encode_action.mlp1.2.weight'].T + WA['encode_action.mlp1.2.bias']
    m = h1.sum(0) / np.sqrt(n) + WA['encode_action.size.weight'][:, 0] * np.log1p(n) + WA['encode_action.size.bias']
    hid = gelu(m @ WA['encode_action.mlp2.0.weight'].T + WA['encode_action.mlp2.0.bias'])
    s = hid @ WA['encode_action.mlp2.2.weight'].T + WA['encode_action.mlp2.2.bias']
    mu = s.mean(); sd = np.sqrt(((s - mu)**2).mean() + 1e-5)
    return m, hid, s, (s - mu) / sd * WA['encode_action.norm.weight'] + WA['encode_action.norm.bias']
for args in [(2.0, 0.1), (0.0, 0.0), (6.0, 0.3)]:
    ur = np.asarray(u_rec(*args), dtype=np.float64).ravel()
    ok = None
    for cI, cN in [('insulin', 'norepinephrine'), (VOCAB['codes'].index('insulin') if isinstance(VOCAB['codes'], list) else None, None)]:
        try:
            V = np.stack([np.asarray(datom(cI, args[0]), dtype=np.float64).ravel(), np.asarray(datom(cN if cN else VOCAB['codes'].index('norepinephrine'), args[1]), dtype=np.float64).ravel()])
            u_my = enc_int(V)[3]; ok = (cI, np.abs(u_my - ur).max()); break
        except Exception as ex: print('datom try fail', cI, ex)
    print(args, 'u_rec shape', ur.shape, 'match', ok, 'setenc_my diff', np.abs(np.asarray(setenc_my(V, 'encode_action'), dtype=np.float64).ravel() - ur).max() if ok else None)
```