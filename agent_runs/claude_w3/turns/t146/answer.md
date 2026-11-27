`enc_bundle` returns e and u (192 numbers each) plus two scalars, and the bundle count matches the saved states (54 for stay 2020000112). I'll now rebuild the z posterior mean for bundles 1-53 of that stay, starting from the saved previous state, in float64 with fresh helpers. The context is out_prev + out_norm(mean of time-flowed slots) + gap(log dt / 10). A relative error near 1e-4 validates the encoders and context; a large error means one of them is wrong.

```python
import torch, torch.nn.functional as Fn
Wd = lambda n: torch.tensor(np.asarray(load_original(n)), dtype=torch.float64)
def LNd(x, p): return Fn.layer_norm(x, (x.shape[-1],), Wd(p+'.weight'), Wd(p+'.bias'), 1e-5)
def LIN(x, p, b=True): return x @ Wd(p+'.weight').T + (Wd(p+'.bias') if b else 0)
def MLP(x, p): return LIN(Fn.gelu(LIN(x, p+'.0')), p+'.2')
def zpost_test(sid, dt_scale=3600.0):
    m = F4['stay_id']==sid; X_ = torch.tensor(F4['x'][m], dtype=torch.float64); hr_ = F4['hour'][m]
    B = stay_bundles(sid); errs = []
    for k in range(1, len(B)):
        e, u = [torch.tensor(np.asarray(t), dtype=torch.float64) for t in enc_bundle(B[k])[:2]]
        dt = (hr_[k]-hr_[k-1])*dt_scale
        hp = X_[k-1,192:960].reshape(4,192) * torch.exp(-dt*torch.exp(Wd('core.flow.a')))
        ctx = X_[k-1,:192] + LNd(hp.mean(0), 'core.out_norm') + MLP(torch.tensor([np.log(dt)/10.]), 'gap')
        zm = MLP(torch.cat([ctx, e, u]), 'z_post')[:32]
        errs.append(((zm - X_[k,960:]).norm()/X_[k,960:].norm()).item())
    return np.round(np.percentile(errs, [50, 90, 100]), 4)
print(zpost_test(s0_))
```