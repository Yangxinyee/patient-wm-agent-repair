Bundle 0 (a bare hospital_admission at −2 h) gives exactly the same state in all 100 stays. That makes it a fixed point to test a CPU copy against: it depends only on the initial state, the atom embedding and the encoders.

The bin tables also matter for the edit:
- The insulin rate z-score, (rate − 1.5)/1.0, is linear in rate and clipped only at ±10. So a doubled dose stays inside the linear range of the numeric embedding.
- The rate bins saturate: everything above 5 units/h falls in bin 16.
- Norepinephrine rate z is (rate − 0.47)/0.6.

So a linear dose signal is available through `embed.numeric`.

Plan for the copy, step 1: implement the atom embedding, the DeepSets encoder and `z_post`. I will then search a small grid of unknowns to match bundle 0's posterior mean z:
- activation function;
- whether the absent-bin rows, the categorical rows and `embed.kq` are added;
- the initial state and the gap input for the first bundle.

A match within about 1e-4 confirms those conventions. If nothing matches, I learn which assumption is wrong and whether a copy is feasible.

```python
import torch, itertools
torch.set_grad_enabled(False)
W = {nm: torch.tensor(load_original(nm)) for nm in NAMES}
F = torch.nn.functional
acts = {'gelu': F.gelu, 'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'), 'silu': F.silu, 'relu': F.relu}
def lin(x, p, bias=True): return x @ W[p+'.weight'].T + (W[p+'.bias'] if bias and (p+'.bias') in W else 0)
def ln(x, p, eps=1e-5): return F.layer_norm(x, (x.shape[-1],), W[p+'.weight'], W[p+'.bias'], eps)
def mlp(x, p, act): return lin(act(lin(x, p+'.0')), p+'.2')
def atom_emb(code, kind, ekind, opts, vb=0, db=0, rb=0, num=(0,0,0,0,0,0)):
    e_ = W['embed.code.weight'][code] + W['embed.kind.weight'][kind] + W['embed.ekind.weight'][ekind]
    if opts.get('bins0', True): e_ = e_ + W['embed.bins.value.weight'][vb] + W['embed.bins.dose.weight'][db] + W['embed.bins.rate.weight'][rb]
    if opts.get('kq') is not None: e_ = e_ + W['embed.kq.weight'][opts['kq']]
    if opts.get('cats', False):
        for t in ['action','discharged_to','rate_unit','route','status','unit']: e_ = e_ + W[f'embed.categorical.{t}.weight'][0]
        e_ = e_ + W['embed.text.weight'][0]
    e_ = e_ + W['embed.numeric.weight'] @ torch.tensor(num, dtype=torch.float32) + W['embed.numeric.bias']
    return e_
def set_enc(atoms, p, act):
    n = atoms.shape[0]
    s_ = mlp(atoms, p+'.mlp1', act).sum(0) / np.sqrt(n) + lin(torch.tensor([np.log1p(n)], dtype=torch.float32), p+'.size')
    return ln(mlp(s_, p+'.mlp2', act), p+'.norm')
zt = torch.tensor(x0[0,960:992])
best = []
for an, act in acts.items():
  for kq in [None,0,1,2,3]:
    for cats in [False, True]:
      for bins0 in [True, False]:
        opts = dict(kq=kq, cats=cats, bins0=bins0)
        e0 = set_enc(atom_emb(11, 0, 3, opts)[None], 'encode', act)
        for hinit in ['zero','slots']:
          h0 = torch.zeros(4,192) if hinit=='zero' else W['core.slots']
          for oinit in ['zero','norm']:
            o0 = torch.zeros(192) if oinit=='zero' else ln(h0.mean(0), 'core.out_norm')
            for gin in [None, 0.0, 1.0, np.log(3600)/10, np.log(7200)/10]:
              ctx = o0 + ln(h0.mean(0), 'core.out_norm') + (0 if gin is None else mlp(torch.tensor([gin],dtype=torch.float32), 'gap', act))
              zo = lin(act(lin(torch.cat([ctx, e0, torch.zeros(192)]), 'z_post.0')), 'z_post.2')
              err = (zo[:32]-zt).abs().max().item()
              best.append((err, an, kq, cats, bins0, hinit, oinit, gin))
best.sort(key=lambda t: t[0])
for b_ in best[:8]: print(b_)
print('target z0[:6]', np.round(x0[0,960:966],4))
```