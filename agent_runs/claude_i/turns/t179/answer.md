`atoms_by_bundle` maps (stay, minute) to a list of atom dicts. The precomputed `ctx` tensor matches the 5156 state bundles. That is enough to rebuild the encoder cleanly from the recipe in my notes:
- start atoms use kq=1 and `categorical.action[1]`;
- end atoms carry only code, kind, event kind and the numeric bias, and are routed to e;
- empty sets are encoded rather than zeroed;
- GELU is used throughout.

I'll write it to /work/enc2.py and check it by recomputing the z posterior mean from (ctx, e, u) against the tool's z for all 5156 bundles. A near-zero error confirms that I can compute u for arbitrary edited doses. That ability is the basis for designing a dose-deviation readout.

```python
code_enc2 = r'''
import numpy as np, torch
import torch.nn.functional as F_
EK2 = {c["name"]: {"measurement":1,"drug_admin":2,"visit":3,"visit_detail":4}.get(c.get("event_kind"),0) for c in VOCAB["codes"]}
UNIT2 = {'glucose_finger':0,'glucose_lab':5,'map_arterial':2,'map_cuff':2,'insulin':3,'norepinephrine':4}
RUNIT2 = {'insulin':2,'norepinephrine':3}
def T64(n): return load(n).double()
def gelu_(x): return F_.gelu(x)
def emb_atoms(atoms):
    W = {k: T64(k) for k in ['embed.code.weight','embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.categorical.unit.weight',
         'embed.categorical.rate_unit.weight','embed.categorical.action.weight','embed.bins.value.weight','embed.bins.rate.weight','embed.bins.dose.weight','embed.numeric.weight','embed.numeric.bias']}
    V = torch.zeros((len(atoms),192), dtype=torch.float64)
    for i,a in enumerate(atoms):
        c = a['code']; kind = a.get('kind',0)
        v = W['embed.code.weight'][CID[c]] + W['embed.kind.weight'][kind] + W['embed.ekind.weight'][EK2[c]]
        num = torch.zeros(6, dtype=torch.float64)
        if c in ('glucose_finger','glucose_lab','map_arterial','map_cuff'):
            v = v + W['embed.kq.weight'][0] + W['embed.categorical.unit.weight'][UNIT2[c]]
            num[0] = float(zsc(c,'value',a['value'])); num[1] = 1
            v = v + W['embed.bins.value.weight'][int(fbin(c,'value',np.array([a['value']]))[0])]
            v = v + W['embed.numeric.weight'] @ num
        elif c in ('insulin','norepinephrine') and kind==0:
            v = v + W['embed.kq.weight'][1] + W['embed.categorical.action.weight'][1] + W['embed.categorical.unit.weight'][UNIT2[c]] + W['embed.categorical.rate_unit.weight'][RUNIT2[c]]
            num[2] = float(zsc(c,'dose',a['dose'])); num[3]=1; num[4] = float(zsc(c,'rate',a['rate'])); num[5]=1
            v = v + W['embed.bins.dose.weight'][int(fbin(c,'dose',np.array([a['dose']]))[0])] + W['embed.bins.rate.weight'][int(fbin(c,'rate',np.array([a['rate']]))[0])]
            v = v + W['embed.numeric.weight'] @ num
        else:
            v = v + W['embed.kq.weight'][0]
        V[i] = v + W['embed.numeric.bias']
    return V
def set_enc2(V, pre):
    n = V.shape[0]
    if n>0:
        h1 = gelu_(V @ T64(pre+'.mlp1.0.weight').T + T64(pre+'.mlp1.0.bias')) @ T64(pre+'.mlp1.2.weight').T + T64(pre+'.mlp1.2.bias')
        s = h1.sum(0)/np.sqrt(n)
    else:
        s = torch.zeros(192, dtype=torch.float64)
    s = s + T64(pre+'.size.weight')[:,0]*np.log1p(n) + T64(pre+'.size.bias')
    m = gelu_(s @ T64(pre+'.mlp2.0.weight').T + T64(pre+'.mlp2.0.bias')) @ T64(pre+'.mlp2.2.weight').T + T64(pre+'.mlp2.2.bias')
    return F_.layer_norm(m, (192,), T64(pre+'.norm.weight'), T64(pre+'.norm.bias'), 1e-5)
def eu2(atoms):
    acts = [a for a in atoms if a['code'] in ('insulin','norepinephrine') and a.get('kind',0)==0]
    obs = [a for a in atoms if not (a['code'] in ('insulin','norepinephrine') and a.get('kind',0)==0)]
    e = set_enc2(emb_atoms(obs), 'encode'); u = set_enc2(emb_atoms(acts), 'encode_action')
    return e, u
def zpost2(ctx, e, u):
    x = torch.cat([ctx, e, u], -1)
    h = gelu_(x @ T64('z_post.0.weight').T + T64('z_post.0.bias'))
    return h @ T64('z_post.2.weight').T + T64('z_post.2.bias'), h
'''
open('/work/enc2.py','w').write(code_enc2); exec(code_enc2)
rowkey = {(int(s), int(round(h*60))): i for i,(s,h) in enumerate(zip(S3, hr_))}
errs=[]; Eall=np.zeros((5156,192)); Uall=np.zeros((5156,192))
for key,atoms in atoms_by_bundle.items():
    i = rowkey.get((int(key[0]), int(key[1])))
    if i is None: continue
    e,u = eu2(atoms); Eall[i]=e.numpy(); Uall[i]=u.numpy()
zz2,_ = zpost2(ctx.double(), torch.tensor(Eall), torch.tensor(Uall))
d_ = (zz2[:,:32].numpy()-X3[:,960:992])
print('n', len(atoms_by_bundle), 'z err median', np.median(np.abs(d_)).round(5), '90%', np.quantile(np.abs(d_).max(1),0.9).round(4), 'max', np.abs(d_).max().round(3))
```