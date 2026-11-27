The config gives 4 attention heads, FFN multiplier 4, one prelude layer, two recurrent layers, one coda layer and a memory of 128. The bundles of each tuning stay are cached in `BUN` (time in minutes, list of events), and `bundle_vecs` gives the bundle's encodings.

Next I'll write the core step (time flow, prelude over [h+slots, e, u, z_in(z)], two recurrent layers with cross-attention to memory and age keys, the gated h update, coda, out_norm). I'll check it against the GPU states, feeding in the GPU's previous state and z. A few details are not in the documentation: how age enters the keys, what unit age is in, and whether the memory includes the current bundle. I'll try the plausible variants and keep whichever reproduces `out` and h to about 1e-5. If none does, I'll look at which part breaks.

```python
import torch, math
import torch.nn.functional as Fn
W = Wt
def linp(x, p, bias=True):
    y = x @ W[p+'.weight'].T
    if bias and (p+'.bias') in W: y = y + W[p+'.bias']
    return y
def lnp(x, p, eps=1e-5): return Fn.layer_norm(x, (x.shape[-1],), W[p+'.weight'], W[p+'.bias'], eps)
def mha(xq, xkv, p, kadd=None, kpre=None):
    nh, dh = 4, 48
    q = xq @ W[p+'.q.weight'].T
    kv = xkv @ W[p+'.kv.weight'].T
    k, v = kv[:, :192], kv[:, 192:]
    if kpre is not None: k = (xkv + kpre) @ W[p+'.kv.weight'][:192].T
    if kadd is not None: k = k + kadd
    q = q.view(-1,nh,dh).transpose(0,1); k = k.view(-1,nh,dh).transpose(0,1); v = v.view(-1,nh,dh).transpose(0,1)
    a = torch.softmax(q @ k.transpose(1,2)/math.sqrt(dh), -1)
    return (a @ v).transpose(0,1).reshape(-1,192) @ W[p+'.out.weight'].T
def ffn(x, p): return linp(Fn.gelu(linp(x, p+'.ffn.0')), p+'.ffn.2')
def tlayer(x, p):
    n1 = lnp(x, p+'.norm1'); x = x + mha(n1, n1, p+'.attn')
    return x + ffn(lnp(x, p+'.norm3'), p)
def block(s, Mem, agek, p, mode):
    n1 = lnp(s, p+'.norm1'); s = s + mha(n1, n1, p+'.attn')
    n2 = lnp(s, p+'.norm2')
    s = s + (mha(n2, Mem, p+'.cross', kadd=agek) if mode=='post' else mha(n2, Mem, p+'.cross', kpre=agek))
    return s + ffn(lnp(s, p+'.norm3'), p)
def step(out_prev, h_prev, dt_s, e_k, u_k, z_k, Mem, ages, mode='post'):
    hf = h_prev * torch.exp(-dt_s*torch.exp(W['core.flow.a']))
    toks = torch.cat([hf + W['core.slots'], e_k[None], u_k[None], linp(z_k,'z_in')[None]], 0)
    E = lnp(tlayer(toks,'core.prelude.0')[:4], 'core.prelude_norm')
    s = hf + E
    agek = linp(Fn.gelu(linp(torch.log1p(ages)[:,None], 'core.age.0')), 'core.age.2')
    for p in ['core.blocks.0.0','core.blocks.0.1']: s = block(s, Mem, agek, p, mode)
    sp = Fn.softplus(W['core.delta_in']); abar = torch.exp(-sp*torch.exp(W['core.a_in']))
    hn = abar*hf + sp*linp(E,'core.b_in',bias=False) + (s - (hf+E))
    y = tlayer(linp(hn,'core.c_out',bias=False), 'core.coda.0')
    return lnp(y.mean(0),'core.out_norm'), y
st0 = list(BUN.keys())[0]; bl = BUN[st0]; idx = np.where(sid_==st0)[0]
print(len(bl), len(idx), type(ROWS[st0]))
r_ = bundle_vecs(bl[0][1]); print(type(r_), [getattr(t,'shape',None) for t in (r_ if isinstance(r_,(tuple,list)) else [r_])])
Xt = torch.tensor(x)
def run_variant(mode, age_unit, mem_cur, kmax=12):
    errs = []
    Es = [bundle_vecs(b[1]) for b in bl]
    times = np.array([b[0] for b in bl], float)*60.0
    for k in range(1, min(kmax, len(bl))):
        out_p = Xt[idx[k-1], :192]; h_p = Xt[idx[k-1], 192:960].view(4,192); z_k = Xt[idx[k], 960:992]
        e_k, u_k = Es[k][0], Es[k][1]
        js = list(range(max(0,k-128+ (1 if mem_cur else 0)), k + (1 if mem_cur else 0)))
        Mem = torch.stack([Es[j][0] for j in js]); ages = torch.tensor((times[k]-times[js])/age_unit, dtype=torch.float32)
        o_, y_ = step(out_p, h_p, float(times[k]-times[k-1]), e_k, u_k, z_k, Mem, ages, mode)
        errs.append(max((o_-Xt[idx[k],:192]).abs().max().item(), (y_.reshape(-1)-Xt[idx[k],192:960]).abs().max().item()))
    return np.round(errs, 5)
for mode in ['post','pre']:
    for au in [1.0, 60.0, 3600.0]:
        for mc in [True, False]:
            try: print(mode, au, mc, run_variant(mode, au, mc))
            except Exception as ex_: print(mode, au, mc, 'ERR', ex_)
```