The timing arrays are what I need:
- `min_` is minutes from ICU admission. The hospital-admission bundle is at -120 min, and drug records sit 5 minutes after each hourly reading.
- `dt_s` is seconds since the previous bundle.
- `prev_ok` marks bundles that have a previous bundle in the same stay.

The tool's h-slot norms are large (~556) and `out` has norm ~14, which fits h being unnormalized and `out` being a LayerNorm output.

Next I will implement one core step in torch (time flow → prelude → two recurrent blocks with cross-attention to the memory → state mix → c_out → coda → out_norm). I will check it one step ahead against the tool's states, using the tool's own previous state and z. I try a few variants for the open details (the memory age unit, whether the memory includes the current bundle, and how age enters the keys) on the first ~250 checkable bundles.

The outcome I am looking for: one variant gives near-zero relative error in h and out (below 1e-3). If none does, the error pattern will show which part is wrong, for example a large error only once memory entries exist.

```python
import torch.nn.functional as Fnn
def LNw(x, pre): return Fnn.layer_norm(x, (192,), WT[pre+'.weight'], WT[pre+'.bias'], 1e-5)
def linw(x, pre, bias=True):
    y = x @ WT[pre+'.weight'].T
    return y + WT[pre+'.bias'] if bias else y
def attnw(xq, xkv, pre, kadd=None, kin=None):
    Wkv = WT[pre+'.kv.weight']; q = xq @ WT[pre+'.q.weight'].T
    if kin is not None: k = (xkv + kin) @ Wkv[:192].T; v = xkv @ Wkv[192:].T
    else: kv = xkv @ Wkv.T; k, v = kv[..., :192], kv[..., 192:]
    if kadd is not None: k = k + kadd
    B, Tq, _ = q.shape; Tk = k.shape[1]
    q = q.view(B,Tq,4,48).transpose(1,2); k = k.reshape(B,Tk,4,48).transpose(1,2); v = v.reshape(B,Tk,4,48).transpose(1,2)
    a = torch.softmax(q @ k.transpose(-1,-2)/np.sqrt(48), -1)
    return (a @ v).transpose(1,2).reshape(B,Tq,192) @ WT[pre+'.out.weight'].T
def ffnw(x, pre): return linw(Fnn.gelu(linw(x, pre+'.0')), pre+'.2')
def age_emb(age): return linw(Fnn.gelu(linw(torch.log1p(age)[...,None], 'core.age.0')), 'core.age.2')
def core_step(hf, e, u, z, M, ages, V):
    zt = linw(z, 'z_in')
    tok = torch.cat([hf + WT['core.slots'], e[None], u[None], zt[None]], 0)[None]
    xn = LNw(tok, 'core.prelude.0.norm1'); x = tok + attnw(xn, xn, 'core.prelude.0.attn')
    x = x + ffnw(LNw(x, 'core.prelude.0.norm3'), 'core.prelude.0.ffn')
    E = LNw(x[0,:4], 'core.prelude_norm')
    x = (hf + E)[None]
    for l in ['core.blocks.0.0', 'core.blocks.0.1']:
        xn = LNw(x, l+'.norm1'); x = x + attnw(xn, xn, l+'.attn')
        if M is not None and M.shape[0] > 0:
            xn = LNw(x, l+'.norm2'); ae = age_emb(ages)[None]; Mm = M[None]
            if V['age'] == 'kadd': x = x + attnw(xn, Mm, l+'.cross', kadd=ae)
            elif V['age'] == 'kin': x = x + attnw(xn, Mm, l+'.cross', kin=ae)
            else: x = x + attnw(xn, Mm + ae, l+'.cross')
        x = x + ffnw(LNw(x, l+'.norm3'), l+'.ffn')
    ub = x[0]
    sp = Fnn.softplus(WT['core.delta_in']); abar = torch.exp(-sp*torch.exp(WT['core.a_in']))
    hp = abar*hf + sp*(E @ WT['core.b_in.weight'].T) + (ub - (hf + E))
    x = (hp @ WT['core.c_out.weight'].T)[None]
    xn = LNw(x, 'core.coda.0.norm1'); x = x + attnw(xn, xn, 'core.coda.0.attn')
    x = x + ffnw(LNw(x, 'core.coda.0.norm3'), 'core.coda.0.ffn')
    return x[0], LNw(x[0].mean(0), 'core.out_norm')
Xt = torch.tensor(X3, dtype=torch.float32); OUTt = Xt[:, :192]; HHt = Xt[:, 192:960].reshape(-1,4,192); ZZt = Xt[:, 960:]
Ee, Uu = euF['e'].float(), euF['u'].float(); tmin = torch.tensor(np.asarray(min_), dtype=torch.float32); dts = torch.tensor(np.asarray(dt_s), dtype=torch.float32)
starts = np.r_[0, np.where(S3[1:] != S3[:-1])[0] + 1]
first_of = np.zeros(NB, int)
for i_, s0_ in enumerate(starts): first_of[s0_: (starts[i_+1] if i_+1 < len(starts) else NB)] = s0_
def check(V, kmax=250):
    errs = []
    for k_ in range(1, NB):
        if not prev_ok[k_]: continue
        if len(errs) >= kmax: break
        hf = HHt[k_-1] * torch.exp(-dts[k_]*torch.exp(WT['core.flow.a']))
        j0 = first_of[k_]; jend = k_+1 if V['incl'] else k_
        idx_m = np.arange(max(j0, jend-128), jend)
        ages = (tmin[k_] - tmin[idx_m]) * {'s':60.,'min':1.,'h':1/60.}[V['unit']]
        hn, on = core_step(hf, Ee[k_], Uu[k_], ZZt[k_], Ee[idx_m], ages, V)
        errs.append([((hn-HHt[k_]).norm()/HHt[k_].norm()).item(), ((on-OUTt[k_]).norm()/OUTt[k_].norm()).item()])
    return np.median(np.array(errs), 0), np.max(np.array(errs), 0)
for V in [dict(incl=0,unit='s',age='kadd'), dict(incl=1,unit='s',age='kadd'), dict(incl=0,unit='h',age='kadd'), dict(incl=1,unit='h',age='kadd'),
          dict(incl=0,unit='s',age='kin'), dict(incl=1,unit='s',age='kin'), dict(incl=0,unit='s',age='both'), dict(incl=1,unit='s',age='both')]:
    print(V, [np.round(a_,4) for a_ in check(V)])
```