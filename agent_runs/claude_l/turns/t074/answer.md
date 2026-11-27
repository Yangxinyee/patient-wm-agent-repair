The bundle format is clear. Each bundle is a time in hours plus a list of atom dicts. Infusion END atoms (kind 2) carry the record's rate and dose and share a bundle with the next record's start. The gap feature is log(dt in seconds)/10. `h` has a large scale (±100), and `out` is layer-normalised.

Next I will write a clean, self-contained implementation of the full core step into /work/wm.py, so it survives restarts. I will test it one step at a time: start from the tool's state after bundle k−1 and compare with the tool's state after bundle k.

I will try a few variants for the unknowns:
- whether the memory M includes the current bundle;
- whether age is in seconds or hours;
- whether age is added to the keys only, or to M before the key/value projection;
- the absent u-token as a zero vector or masked, and the absent e as the empty-set encoding.

One variant giving near-zero error for `out` and `h` would confirm the reconstruction. If all variants fail, the error pattern (which tensor or stage diverges first) will point to the wrong component.

```python
code = r'''
import torch, numpy as np, math
import torch.nn.functional as F
T = torch.float64
ACT = F.silu
def getW(loader, names):
    return {n: torch.tensor(np.asarray(loader(n)), dtype=T) for n in names}
def lin(W, x, n, bias=True):
    y = x @ W[n+'.weight'].T
    if bias and (n+'.bias') in W: y = y + W[n+'.bias']
    return y
def mlp(W, x, n): return lin(W, ACT(lin(W, x, n+'.0')), n+'.2')
def lnorm(W, x, n, eps=1e-5): return F.layer_norm(x, (x.shape[-1],), W[n+'.weight'], W[n+'.bias'], eps)
def atom_matrix(W, atoms):
    rows=[]
    for a in atoms:
        v = (W['embed.code.weight'][a['code']] + W['embed.kind.weight'][a['kind']] + W['embed.ekind.weight'][a['ekind']]
             + W['embed.kq.weight'][0] + W['embed.categorical.unit.weight'][a['unit']] + W['embed.categorical.rate_unit.weight'][a['rate_unit']]
             + W['embed.bins.value.weight'][a['vb']] + W['embed.bins.dose.weight'][a['db']] + W['embed.bins.rate.weight'][a['rb']]
             + W['embed.numeric.weight'] @ torch.tensor(a['num'], dtype=T) + W['embed.numeric.bias'])
        rows.append(v)
    return torch.stack(rows)
def set_enc(W, atoms, pre):
    n = len(atoms)
    if n == 0: s = torch.zeros(192, dtype=T)
    else: s = mlp(W, atom_matrix(W, atoms), pre+'.mlp1').sum(0) / math.sqrt(n)
    s = s + lin(W, torch.tensor([math.log(1+n)], dtype=T), pre+'.size')
    return lnorm(W, mlp(W, s, pre+'.mlp2'), pre+'.norm')
def mha(W, xq, xkv, n, kadd=None, mask=None, H=4):
    q = xq @ W[n+'.q.weight'].T
    kv = xkv @ W[n+'.kv.weight'].T
    k, v = kv[:, :192], kv[:, 192:]
    if kadd is not None: k = k + kadd
    dh = 192//H
    q = q.reshape(-1,H,dh).transpose(0,1); k = k.reshape(-1,H,dh).transpose(0,1); v = v.reshape(-1,H,dh).transpose(0,1)
    att = (q @ k.transpose(1,2)) / math.sqrt(dh)
    if mask is not None: att = att.masked_fill(~mask[None,None,:], float('-inf'))
    o = torch.softmax(att, -1) @ v
    return o.transpose(0,1).reshape(-1,192) @ W[n+'.out.weight'].T
def self_layer(W, x, L, mask=None):
    a = lnorm(W, x, L+'.norm1'); x = x + mha(W, a, a, L+'.attn', mask=mask)
    x = x + mlp(W, lnorm(W, x, L+'.norm3'), L+'.ffn'); return x
'''
open('/work/wm.py','w').write(code)
exec(code)
W0 = getW(load_original, NAMES)
# check atom embedding & encoders vs old verified functions
aa = BUN[2020000000][1][1]
old = atomvec(aa[0]).numpy(); new = atom_matrix(W0,[aa[0]])[0].numpy()
print('atomvec diff', np.abs(old-new).max())
e_old = setenc(aa,'encode',F.silu,F.silu).numpy(); e_new = set_enc(W0,aa,'encode').numpy()
print('setenc diff', np.abs(e_old-e_new).max())
eo = empty_enc('encode',F.silu).numpy(); en = set_enc(W0,[], 'encode').numpy(); print('empty diff',np.abs(eo-en).max())
```