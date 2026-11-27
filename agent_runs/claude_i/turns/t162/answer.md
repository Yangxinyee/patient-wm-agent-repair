The disassembly gives `step_err`'s exact conventions:

- It starts from the tool's state after the previous bundle and applies the time flow with dt in seconds.
- The memory M is the e-vectors of the stay's bundles from max(first bundle, k−127) up to and including bundle k.
- Ages are (t_k − t_j) in minutes × 60, i.e. seconds.
- It calls `core_step(hf, e_k, u_k, z_k, M, ages, V)` with the tool's posterior z.

To simulate stays with altered doses I must compute z myself as z_post([ctx, e, u]) and chain my own states from the start of the stay. Next I write a self-contained driver to /work/sim.py so it survives restarts. It uses my own z_post and ctx and assumes zero initial h/out. I run it from scratch on a few stays and compare with the tool's states.

If the error stays at the 1e-3 level along whole stays, the driver is usable for counterfactual dose runs. If it is large from the first bundle, the initial-state or first-dt convention is wrong.

```python
code_sim = r'''
import numpy as np, torch
import torch.nn.functional as Fnn_
def _W(n): return WT[n]
def zpost_mean(ctx, e, u):
    x = torch.cat([ctx, e, u], -1)
    hdn = Fnn_.gelu(x @ WT['z_post.0.weight'].T + WT['z_post.0.bias'])
    return (hdn @ WT['z_post.2.weight'].T + WT['z_post.2.bias'])[..., :32]
def gapf(dt_s):
    g = torch.log(torch.clamp(torch.as_tensor(dt_s, dtype=WT['gap.0.weight'].dtype), min=1.0)).reshape(-1,1)/10.0
    return (Fnn_.gelu(g @ WT['gap.0.weight'].T + WT['gap.0.bias']) @ WT['gap.2.weight'].T + WT['gap.2.bias'])
def ctx_from(out_prev, hf, dt_s):
    return out_prev + Fnn_.layer_norm(hf.mean(-2), (192,), WT['core.out_norm.weight'], WT['core.out_norm.bias'], 1e-5) + gapf(dt_s).reshape(out_prev.shape)
def run_stay(ks, E_, U_, V_={'incl':1,'unit':'s','age':'both'}, h0=None, o0=None, collect=True):
    """ks: global bundle indices of one stay (in order). E_,U_: (N,192) tensors for those global indices."""
    dt_ = WT['core.flow.a'].dtype
    h = torch.zeros(4,192, dtype=dt_) if h0 is None else h0.clone()
    o = torch.zeros(192, dtype=dt_) if o0 is None else o0.clone()
    j0 = ks[0]; Hs=[]; Os=[]; Zs=[]; CT=[]
    for k in ks:
        hf = h * torch.exp(-dts[k] * torch.exp(WT['core.flow.a']))
        ctx = ctx_from(o, hf, dts[k])
        z = zpost_mean(ctx, E_[k], U_[k])
        idx_m = np.arange(max(j0, k+1-128), k+1)
        ages = (tmin[k] - tmin[idx_m]) * 60.0
        h, o = core_step(hf, E_[k], U_[k], z, E_[idx_m], ages, V_)
        if collect: Hs.append(h); Os.append(o); Zs.append(z); CT.append(ctx)
    return Hs, Os, Zs, CT
'''
open('/work/sim.py','w').write(code_sim)
exec(code_sim)
print({k:SHAPES[k] for k in NAMES if k.startswith('gap')})
print("WT n", len(WT), "dtype", WT['core.flow.a'].dtype, type(Ee), Ee.dtype if hasattr(Ee,'dtype') else None)
# stays in fwd_3 ordering
starts=[i for i in range(len(first_of)) if first_of[i]==i]
print("n stays", len(starts), "dts first bundles", [float(dts[s]) for s in starts[:5]])
for s_ in starts[:3]:
    ks=list(range(s_, (starts[starts.index(s_)+1] if starts.index(s_)+1<len(starts) else len(first_of))))
    Hs,Os,Zs,CT=run_stay(ks, Ee, Uu)
    errs=[float((Hs[i]-torch.as_tensor(HHt[k]).to(Hs[i].dtype)).norm()/torch.as_tensor(HHt[k]).norm()) for i,k in enumerate(ks)]
    zerr=[float((Zs[i]-torch.as_tensor(ZZt[k]).to(Zs[i].dtype)).norm()/ (torch.as_tensor(ZZt[k]).norm()+1e-9)) for i,k in enumerate(ks)]
    print("stay idx",s_,"len",len(ks),"h relerr first/median/max %.2e %.2e %.2e"%(errs[0],np.median(errs),max(errs)), " z relerr med/max %.2e %.2e"%(np.median(zerr),max(zerr)))
```