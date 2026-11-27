The tokenisation now matches all 100 stays exactly: bundle times and atom counts, including END atoms for infusions and for the two admission records at stay end.

Next I'll replicate the first part of a step: atom embedding, the DeepSets encoder, the context vector and the z posterior mean. I'll check it against the model's reported z on bundles holding only measurements, where the action vector is zero. That isolates the measurement embedding, the encoder, the time-flow decay of h and the gap term. The previous out and h come straight from the states file.

Unknowns are which unit-table rows go with mmHg and mg/dL, and the activation (GELU first). I'll try each unit mapping. A match to about 1e-4 would confirm those pieces. If every variant fails, the error is in the context or gap formula.

```python
import torch, math, itertools
torch.set_default_dtype(torch.float64)
Wt = {n_: torch.tensor(load_original(n_), dtype=torch.float64) for n_ in NAMES}
F = torch.nn.functional
code_id = {c['name']: c['id'] for c in VOCAB['codes']}
NUM = {(d_['code'], d_['attribute']): d_ for d_ in VOCAB['numeric']}
def zsc(code, attr, x):
    d_ = NUM[(code, attr)]; return float(np.clip((x-d_['median'])/d_['iqr'], -10, 10))
def fbin(code, attr, x):
    e_ = np.array(NUM[(code, attr)]['fine_bin_edges']); return int(1 + (e_ < x).sum())
ACT = F.gelu
def lin(x, p, bias=True):
    y = x @ Wt[p+'.weight'].T
    return y + Wt[p+'.bias'] if bias and (p+'.bias') in Wt else y
def mlp(x, p): return lin(ACT(lin(x, p+'.0')), p+'.2')
def ln(x, p, eps=1e-5): return F.layer_norm(x, (x.shape[-1],), Wt[p+'.weight'], Wt[p+'.bias'], eps)
UNIT = {'mmHg':2, 'mg/dL':3}
def meas_atom(code, val):
    x = Wt['embed.code.weight'][code_id[code]] + Wt['embed.kind.weight'][0] + Wt['embed.ekind.weight'][1] + Wt['embed.kq.weight'][0]
    unit = 'mmHg' if code.startswith('map') else 'mg/dL'
    x = x + Wt['embed.categorical.unit.weight'][UNIT[unit]]
    x = x + Wt['embed.bins.value.weight'][fbin(code,'value',val)]
    num = torch.tensor([zsc(code,'value',val), 1., 0., 0., 0., 0.])
    return x + Wt['embed.numeric.weight'] @ num + Wt['embed.numeric.bias']
def encode(atoms, p='encode'):
    A = torch.stack(atoms); n = len(atoms)
    s_ = mlp(A, p+'.mlp1').sum(0)/math.sqrt(n) + lin(torch.tensor([math.log(1+n)]), p+'.size')
    return ln(mlp(s_, p+'.mlp2'), p+'.norm')
Xs = torch.tensor(S['x'])
def ctx_of(prev_row, dt):
    out_p = Xs[prev_row, :192]; h_p = Xs[prev_row, 192:960].reshape(4,192)
    dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
    hm = (h_p*dec).mean(0)
    return out_p + ln(hm, 'core.out_norm') + mlp(torch.tensor([math.log(dt)/10]), 'gap')
# collect measurement-only bundles
tests = []
stays_u = list(dict.fromkeys(sid.tolist()))
row = 0
for s_ in stays_u[:20]:
    bb = build_bundles(tu[tu.stay_id==s_]); keys = sorted(bb.keys())
    for i_, k_ in enumerate(keys):
        if i_>0 and all(k2=='world' and e2 not in ('hospital_admission','icu_admission') for e2,k2,_ in bb[k_]):
            tests.append((row+i_, row+i_-1, (k_-keys[i_-1])*60.0, [(e2, r2.value) for e2,k2,r2 in bb[k_]]))
    row += len(keys)
print(len(tests))
def z_err(units_map, ntest=60):
    global UNIT; UNIT = units_map; errs=[]
    for r_, rp, dt, ms_ in tests[:ntest]:
        e_ = encode([meas_atom(c, v) for c, v in ms_])
        ctx = ctx_of(rp, dt)
        zp = mlp(torch.cat([ctx, e_, torch.zeros(192)]), 'z_post')[:32]
        errs.append((zp - Xs[r_, 960:992]).abs().max().item())
    return np.median(errs), np.max(errs)
for a_, b_ in itertools.permutations([2,3,4,5], 2):
    print(a_, b_, z_err({'mmHg':a_, 'mg/dL':b_}, 30))
```