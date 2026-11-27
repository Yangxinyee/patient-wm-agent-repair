
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

def cross_mem(W, x, L, Mt, ages, age_unit, age_mode):
    if Mt is None or Mt.shape[0]==0: return torch.zeros_like(x)
    a = ages/3600.0 if age_unit=='h' else ages
    ae = mlp(W, torch.log1p(a)[:,None], 'core.age')
    xq = lnorm(W, x, L+'.norm2')
    if age_mode=='key':
        return mha(W, xq, Mt, L+'.cross', kadd=ae)
    else:
        return mha(W, xq, Mt+ae, L+'.cross')
def core_step(W, out, h, dt, e, u, z_override=None, Mt=None, ages=None, opt=None):
    o = opt or {}
    hf = h * torch.exp(-dt * torch.exp(W['core.flow.a']))
    g = mlp(W, torch.tensor([math.log(dt)/10], dtype=T), 'gap')
    ctx = out + lnorm(W, hf.mean(0), 'core.out_norm') + g
    zp = mlp(W, torch.cat([ctx, e, u]), 'z_post'); z = zp[:32] if z_override is None else z_override
    tok = torch.cat([hf + W['core.slots'], e[None], u[None], lin(W, z, 'z_in')[None]], 0)
    mask = o.get('pmask')
    tok = self_layer(W, tok, 'core.prelude.0', mask=mask)
    E = lnorm(W, tok[:4], 'core.prelude_norm')
    x = hf + E
    for L in ['core.blocks.0.0','core.blocks.0.1']:
        a_ = lnorm(W, x, L+'.norm1'); x = x + mha(W, a_, a_, L+'.attn')
        x = x + cross_mem(W, x, L, Mt, ages, o.get('age_unit','s'), o.get('age_mode','key'))
        x = x + mlp(W, lnorm(W, x, L+'.norm3'), L+'.ffn')
    sp = F.softplus(W['core.delta_in']); abar = torch.exp(-sp*torch.exp(W['core.a_in']))
    h2 = abar*hf + sp*(E @ W['core.b_in.weight'].T) + (x - (hf + E))
    y = self_layer(W, h2 @ W['core.c_out.weight'].T, 'core.coda.0')
    return lnorm(W, y.mean(0), 'core.out_norm'), y, z, dict(E=E, hf=hf, ctx=ctx)

ACTMAP = {}
_ACTS = {'silu':F.silu,'gelu':F.gelu,'gelu_tanh':(lambda x: F.gelu(x,approximate='tanh')),'relu':F.relu}
def act_for(n):
    for p,a in ACTMAP.items():
        if n.startswith(p): return _ACTS[a]
    return F.silu
def mlp(W, x, n): return lin(W, act_for(n)(lin(W, x, n+'.0')), n+'.2')
