Type index 1 with GELU is the clear winner: the mean absolute error on the 364 first readings is 2.3 mg/dL or mmHg, while every other type gives 26 or more. The remaining error could come from the tool's four random prior draws, since I use the prior mean.

I will test that by averaging probabilities over many prior samples. If the error falls, the replica's structure is right. I will also search the layout of the `last_numeric` register on the readings that have a previous value: which columns hold has-flags, age and z_last_value, and whether `last_status[1]` is added. If the layout is right, the error on those readings should fall to about the same level as for the first readings.

```python
def zsamp_view(K=64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mu, ls = zp_[:, :32], zp_[:, 32:]
    vs = []
    for k in range(K):
        z = mu + torch.exp(ls) * torch.randn(mu.shape, generator=gen, dtype=torch.float64)
        vs.append(out_ + mh_ + gv_ + Hlin(z, 'z_head'))
    return vs
def head_mean_avg(views, typ, act, extra, tb=None):
    P = None
    for v in views:
        _, p = head_mean(v, typ, act, extra, tb)
        P = p if P is None else P + p
    P = P / len(views)
    return (P[:, 1:] * cen_).sum(1), P
vs64 = zsamp_view(64)
mu64, _ = head_mean_avg(vs64, 1, Fn.gelu, ext0)
print('first readings MAE prior-mean %.3f  64-sample %.3f' % (float((head_mean(view0_, 1, Fn.gelu, ext0)[0] - tool_)[m1].abs().mean()), float((mu64 - tool_)[m1].abs().mean())))
# search last_numeric layout on has_last readings (prior-mean z for speed)
m2 = ~m1
LW = HT('last_numeric.weight'); LB = HT('last_numeric.bias')
zl_ = torch.tensor([zs_(e, v) if h else 0.0 for e, v, h in zip(RTb.event, RTb.last_val, RTb.has_last)], dtype=torch.float64)
lb_ = torch.tensor([fb_(e, v) if h else 0 for e, v, h in zip(RTb.event, RTb.last_val, RTb.has_last)])
ag_ = torch.tensor(np.log1p(RTb.age.values) / 10.0)
hl_ = torch.tensor(RTb.has_last.values, dtype=torch.float64)
import itertools
best_ = []
base_ext = LB[None, :] + HT('embed.bins.value.weight')[lb_] + HT('embed.bins.rate.weight')[0]
for ia in range(8):
    for iz in range(8):
        if iz == ia: continue
        rest_ = [i for i in range(8) if i not in (ia, iz)]
        hsets = [(i,) for i in rest_] + list(itertools.combinations(rest_, 2))
        for hs in hsets:
            core_ = base_ext + ag_[:, None] * LW[:, ia] + zl_[:, None] * LW[:, iz] + hl_[:, None] * LW[:, list(hs)].sum(1)
            for st in (0, 1):
                ext_ = core_ + HT('last_status.weight')[st] * hl_[:, None]
                mu_, _ = head_mean(view0_, 1, Fn.gelu, ext_, tb=lb_)
                best_.append((float((mu_ - tool_)[m2].abs().mean()), ia, iz, hs, st))
best_.sort(key=lambda t: t[0])
print('best layouts:', [(round(a, 3), ia, iz, hs, st) for a, ia, iz, hs, st in best_[:5]])
```