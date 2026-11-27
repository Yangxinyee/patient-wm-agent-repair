The code object exposes `pred_mean_s`'s structure:
- input: previous bundle, code, observed value, tool mean, last value and last age;
- view: ctx plus z_head of prior samples;
- q: query_norm of the view plus `type_emb[typ]`;
- with_code: q plus code embedding, `last_numeric` (layout 'A' or 'B'), last value bin, rate bin 0 and `last_status`;
- output: value_bin MLP plus the transition table, softmax, mean over centres.

I'll write my own explicit version to /work/head2.py with the prior mean in place of samples. I'll then search a few `last_numeric` layouts, age units and type indices for the one that matches the tool's factual predicted means in F. A median absolute error of about 1 mg/dL or less, the level of the tool's own sampling noise, would mean I have a usable head. I need it to measure how injected state changes move glucose and MAP predictions.

```python
code_head2 = r'''
import numpy as np, torch
import torch.nn.functional as F_
CENT2 = {k: torch.tensor(v, dtype=torch.float64) for k,v in VOCAB["value_bin_centres"].items()}
def lin2(x, pre): return x @ T64(pre+'.weight').T + (T64(pre+'.bias') if (pre+'.bias') in SHAPES else 0)
def ctx_of_state(out_prev, h_prev, dt_s):
    hf = h_prev*torch.exp(-torch.as_tensor(dt_s,dtype=torch.float64).reshape(-1,1,1)*torch.exp(T64('core.flow.a')))
    return out_prev + F_.layer_norm(hf.mean(1),(192,),T64('core.out_norm.weight'),T64('core.out_norm.bias'),1e-5) + gap2(dt_s)
def zprior2(ctx):
    p = gelu_(lin2(ctx,'z_prior.0')) @ T64('z_prior.2.weight').T + T64('z_prior.2.bias')
    return p[...,:32], p[...,32:]
def value_mean(ctx, codes, lastv, lastage_s, typ=1, lay=0, nz=0, seed=0, extra_view=None):
    """ctx (N,192); codes list of str; lastv (N,) nan if none; lastage_s (N,). returns mean (N,), probs (N,17)"""
    N = ctx.shape[0]; mu, ls = zprior2(ctx)
    if nz==0: Z = mu[None]
    else:
        g = torch.Generator().manual_seed(seed); eps = torch.randn((nz,N,32), generator=g, dtype=torch.float64)
        Z = mu[None] + torch.exp(ls)[None]*eps
    view = ctx[None] + lin2(Z,'z_head')
    if extra_view is not None: view = view + extra_view[None]
    q = F_.layer_norm(view + T64('type_emb.weight')[typ], (192,), T64('query_norm.weight'), T64('query_norm.bias'), 1e-5)
    cid = torch.tensor([CID[c] for c in codes])
    has = torch.tensor(~np.isnan(lastv), dtype=torch.float64)
    zl = torch.tensor([0.0 if np.isnan(v) else float(zsc(c,'value',v)) for c,v in zip(codes,lastv)], dtype=torch.float64)
    ag = torch.tensor(np.where(np.isnan(lastv),0.0,np.log1p(np.nan_to_num(lastage_s))/10.0), dtype=torch.float64)
    feats = torch.zeros((N,8), dtype=torch.float64)
    if lay==0: feats[:,0]=has; feats[:,1]=ag; feats[:,2]=zl
    elif lay==1: feats[:,0]=has; feats[:,1]=ag; feats[:,2]=zl; feats[:,4]=has
    elif lay==2: feats[:,0]=has; feats[:,1]=ag*has; feats[:,2]=zl; feats[:,3]=0; feats[:,5]=0
    lb = torch.tensor([0 if np.isnan(v) else int(fbin(c,'value',np.array([v]))[0]) for c,v in zip(codes,lastv)])
    wc = q + T64('embed.code.weight')[cid][None] + (feats @ T64('last_numeric.weight').T + T64('last_numeric.bias'))[None] \
         + T64('embed.bins.value.weight')[lb][None] + T64('embed.bins.rate.weight')[0][None] + T64('last_status.weight')[0][None]
    lg = gelu_(lin2(wc,'attr_heads.value_bin.0')) @ T64('attr_heads.value_bin.2.weight').T + T64('attr_heads.value_bin.2.bias')
    lg = lg + T64('embed_transition.value_bin.weight')[cid*17+lb][None]
    p = torch.softmax(lg, -1).mean(0)
    cen = torch.stack([CENT2[c] for c in codes])
    if cen.shape[1]==16: m = (p[:,1:]*cen).sum(1)/p[:,1:].sum(1)
    else: m = (p*cen).sum(1)
    return m, p
'''
open('/work/head2.py','w').write(code_head2); exec(code_head2)
print({k: len(v) for k,v in VOCAB['value_bin_centres'].items()})
Fm = F[F.event.isin(['glucose_finger','glucose_lab','map_arterial','map_cuff'])].reset_index(drop=True)
pkF = Fm.pk.values.astype(int); dtF = torch.tensor(Fm.dt.values, dtype=torch.float64)
ctxF = ctx_of_state(X3t[pkF,:192], X3t[pkF,192:960].reshape(-1,4,192), dtF)
lastage = ((Fm.hour - Fm.lasth)*3600).values
for typ in [1,0,2]:
    for lay in [0,1]:
        for unit,fac in [('s',1),('min',1/60),('h',1/3600)]:
            m_,_ = value_mean(ctxF, Fm.event.tolist(), Fm.lastv.values, lastage*fac, typ=typ, lay=lay)
            mad = np.abs(m_.numpy()-Fm.predicted_mean.values)
            print(typ, lay, unit, 'MAD med', np.median(mad).round(3), 'mean', mad.mean().round(3))
```