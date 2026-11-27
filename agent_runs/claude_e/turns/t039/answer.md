All 100 stays match the tool exactly in bundle times and atom counts, so the bundling convention (END atoms at the record end, final admission-END bundle) is right.

Next is the first isolated check of the encoders. For bundle k, I'll take the tool's out and h after bundle k−1 and:
1. apply the time-flow decay over dt;
2. form the context (out + out_norm(mean of slots) + gap(log dt / 10));
3. encode bundle k's measurement atoms and action atoms with my guessed conventions;
4. run `z_post` and compare its mean with the tool's z.

I'll start with measurement-only bundles, which have no action term. That isolates the unit index for mmHg and mg/dL and the activation function. I'll try GELU, SiLU and ReLU, and several unit-index orders. An exact match (error near float precision) for one combination confirms it.

```python
import itertools
def act_fn(name): return {'gelu': Fnn.gelu, 'silu': Fnn.silu, 'relu': Fnn.relu, 'gelu_tanh': lambda x: Fnn.gelu(x, approximate='tanh')}[name]
CFG = dict(act='gelu', unit={'mmHg':2,'mg/dL':3,'units/hour':4,'mcg/kg/min':5}, rate_unit={'units/hour':2,'mcg/kg/min':3}, end_attrs=True, kq=True, empty_e='zero')
code_unit = {c['name']: c.get('unit') for c in V['codes'] if 'unit' in c}
def atom_vec(a, cfg):
    c = a['code']; v = W['embed.code.weight'][code_id[c]] + W['embed.kind.weight'][a['kind']] + W['embed.ekind.weight'][EK[c]]
    if cfg['kq']: v = v + W['embed.kq.weight'][0]
    has_attr = (a['kind']==0) or cfg['end_attrs']
    u = code_unit.get(c)
    if u is not None and has_attr and u in cfg['unit']: v = v + W['embed.categorical.unit.weight'][cfg['unit'][u]]
    if c in ('insulin','norepinephrine') and has_attr: v = v + W['embed.categorical.rate_unit.weight'][cfg['rate_unit'][u]]
    feats = [0.0]*6
    if has_attr:
        if not np.isnan(a['value']):
            v = v + W['embed.bins.value.weight'][fine_bin(c,'value',a['value'])]; feats[0],feats[1] = zsc(c,'value',a['value'])
        if not np.isnan(a['dose']):
            v = v + W['embed.bins.dose.weight'][fine_bin(c,'dose',a['dose'])]; feats[2],feats[3] = zsc(c,'dose',a['dose'])
        if not np.isnan(a['rate']):
            v = v + W['embed.bins.rate.weight'][fine_bin(c,'rate',a['rate'])]; feats[4],feats[5] = zsc(c,'rate',a['rate'])
    v = v + W['embed.numeric.weight'] @ torch.tensor(feats) + W['embed.numeric.bias']
    return v
def mlp(x, p, act): return Fnn.linear(act(Fnn.linear(x, W[p+'.0.weight'], W[p+'.0.bias'])), W[p+'.2.weight'], W[p+'.2.bias'])
def ln(x, p, eps=1e-5): return Fnn.layer_norm(x, (x.shape[-1],), W[p+'.weight'], W[p+'.bias'], eps)
def set_encode(atoms, p, cfg):
    act = act_fn(cfg['act'])
    if len(atoms)==0: return None
    X = torch.stack([atom_vec(a,cfg) for a in atoms]); n = len(atoms)
    s = mlp(X, p+'.mlp1', act).sum(0)/np.sqrt(n) + Fnn.linear(torch.tensor([np.log(1+n)],dtype=torch.float32), W[p+'.size.weight'], W[p+'.size.bias'])
    return ln(mlp(s, p+'.mlp2', act), p+'.norm')
def split_atoms(atoms):
    acts = [a for a in atoms if a['code'] in ('insulin','norepinephrine')]; obs = [a for a in atoms if a['code'] not in ('insulin','norepinephrine')]
    return obs, acts
def zpost_check(sid, cfg, k_list=None, flow_scale=1.0):
    m = np.where(S2['stay_id']==sid)[0]; X = torch.tensor(S2['x'][m]); Bk = Bs[sid]; act = act_fn(cfg['act']); errs=[]
    fa = torch.exp(W['core.flow.a'])
    for k in (k_list if k_list is not None else range(1,len(Bk))):
        out_p = X[k-1,:192]; h_p = X[k-1,192:960].reshape(4,192); dt = (Bk[k][0]-Bk[k-1][0])*3600
        hf = h_p*torch.exp(-dt*fa*flow_scale)
        ctx = out_p + ln(hf.mean(0),'core.out_norm') + mlp(torch.tensor([np.log(dt)/10],dtype=torch.float32),'gap',act)
        obs, acts = split_atoms(Bk[k][1])
        e = set_encode(obs,'encode',cfg); u = set_encode(acts,'encode_action',cfg)
        e = torch.zeros(192) if e is None else e; u = torch.zeros(192) if u is None else u
        zm = mlp(torch.cat([ctx,e,u]),'z_post',act)[:32]
        errs.append((k, len(obs), len(acts), float((zm-X[k,960:]).abs().max()), float(X[k,960:].abs().max())))
    return errs
for actn in ['gelu','silu','relu']:
    cfg = dict(CFG, act=actn); e = zpost_check(2020000000, cfg, k_list=[4,6,8,10])
    print(actn, [(k,no,na,round(err,4),round(mx,3)) for k,no,na,err,mx in e])
```