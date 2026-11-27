The row norms tell me which table rows were used in training. Untrained rows sit at about 0.277 (the initialisation std 0.02 × √192):
- **Never used:** kind 1 (REVEAL); kq rows 1–3; the second row of text, status, route, action and discharged_to; unit row 1; rate_unit row 1.
- **Used:** unit rows 2–5 (four units), rate_unit rows 2–3, kq row 0 for all atoms, and ekind rows 1–4.

Dose bins 13–16 are barely trained (norm about 0.3), while rate bins are well trained. So the model encodes doses mainly through rate bins and the numeric z-scores. Remaining unknowns include the unit-row order, the activation function and the `last_numeric` column order.

The very first bundle is the cleanest test: hospital_admission alone, from an empty state. Its z posterior mean depends only on the initial context and one atom's encoding. Next I'll build the basic blocks and compare this bundle's z mean with the tool's under various guesses: activation, initial h/out, the empty-action token, and the gap input.

```python
import torch.nn.functional as F
torch.set_grad_enabled(False)
def lin(x, n, bias=True):
    y = x @ W[n+'.weight'].T
    if bias and (n+'.bias') in W: y = y + W[n+'.bias']
    return y
def LN(x, n, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), W[n+'.weight'], W[n+'.bias'], eps)
ACT = {'gelu': F.gelu, 'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'), 'relu': F.relu, 'silu': F.silu}
def mlp(x, n, act):
    return lin(ACT[act](lin(x, n+'.0')), n+'.2')
def encode_set(atoms, prefix, act):
    # atoms: (n,192)
    n = atoms.shape[0]
    s = ACT[act](lin(atoms, prefix+'.mlp1.0')); s = lin(s, prefix+'.mlp1.2')
    pooled = s.sum(0)/np.sqrt(n) + lin(torch.tensor([np.log(1+n)], dtype=torch.float32), prefix+'.size')
    y = lin(ACT[act](lin(pooled, prefix+'.mlp2.0')), prefix+'.mlp2.2')
    return LN(y, prefix+'.norm')
X = z['x']; sid_arr = z['stay_id']; hr_arr = z['hour']
ref0 = X[(sid_arr==sid0)][0]
print("ref z after bundle0:", np.round(ref0[960:992][:8],4))
print("ref out after bundle0:", np.round(ref0[:8],4))
# hospital admission atom
def atom_vec(code, kind, ekind, unit=0, rate_unit=0, vbin=0, dbin=0, rbin=0, num=(0,0,0,0,0,0), kq=0):
    v = W['embed.code.weight'][code] + W['embed.kind.weight'][kind] + W['embed.ekind.weight'][ekind] + W['embed.kq.weight'][kq]
    v = v + W['embed.categorical.unit.weight'][unit] + W['embed.categorical.rate_unit.weight'][rate_unit]
    v = v + W['embed.bins.value.weight'][vbin] + W['embed.bins.dose.weight'][dbin] + W['embed.bins.rate.weight'][rbin]
    v = v + W['embed.numeric.weight'] @ torch.tensor(num, dtype=torch.float32) + W['embed.numeric.bias']
    return v
a0 = atom_vec(11, 0, 3)
results=[]
for act in ACT:
    e0 = encode_set(a0[None], 'encode', act)
    u0 = torch.zeros(192)
    for hinit in ['zero','slots']:
        h = torch.zeros(4,192) if hinit=='zero' else W['core.slots'].clone()
        out = torch.zeros(192)
        for gapin in [None, 0.0]:
            g = torch.zeros(192) if gapin is None else mlp(torch.tensor([gapin]), 'gap', act)
            for on in ['ln','none']:
                hm = LN(h.mean(0), 'core.out_norm') if on=='ln' else h.mean(0)
                ctx = out + hm + g
                zp = mlp(torch.cat([ctx, e0, u0]), 'z_post', act)
                err = np.abs(zp[:32].numpy() - ref0[960:992]).max()
                results.append((act, hinit, gapin, on, err))
print(sorted(results, key=lambda r: r[-1])[:8])
```