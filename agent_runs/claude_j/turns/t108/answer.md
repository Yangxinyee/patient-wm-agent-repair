The `last_numeric` layout is found. The has-flag sits in column 0, log(1+age in seconds)/10 in column 3 and z_last_value in column 6, and `last_status` is not added. With that layout the error on the 4,850 readings that have a previous value drops to 0.94, compared with 1.5 or more for the next best layouts. First readings stay at about 2.2 even with 64 z samples, so something in their "no last value" inputs is probably handled differently.

I will test a few variants for first readings: leaving out the `last_numeric` bias or the zero-bin embeddings, and adding `last_status[1]`. I will also check age units and z sampling for the has-last readings. Exactness is not critical, since the edit will be designed on this replica and validated with the tool. Still, a cleaner match makes the hidden-unit analysis more trustworthy.

```python
def ext_for(layout_has=(0,), ia=3, iz=6, age_div=1.0, inc_bias_first=True, inc_bins_first=True, st_first=None):
    ag = torch.tensor(np.log1p(RTb.age.values / age_div) / 10.0)
    e = HT('embed.bins.value.weight')[lb_] + HT('embed.bins.rate.weight')[0] + LB[None, :]
    e = e + ag[:, None] * LW[:, ia] + zl_[:, None] * LW[:, iz] + hl_[:, None] * LW[:, list(layout_has)].sum(1)
    f = (hl_ == 0)
    if not inc_bias_first: e[f] = e[f] - LB
    if not inc_bins_first: e[f] = e[f] - HT('embed.bins.value.weight')[0] - HT('embed.bins.rate.weight')[0]
    if st_first is not None: e[f] = e[f] + HT('last_status.weight')[st_first]
    return e
var_ = {'base': {}, 'noLBfirst': dict(inc_bias_first=False), 'nobinsfirst': dict(inc_bins_first=False),
        'nobothfirst': dict(inc_bias_first=False, inc_bins_first=False), 'st1first': dict(st_first=1),
        'age_min': dict(age_div=60.0), 'age_hr': dict(age_div=3600.0)}
for nm_v, kw_v in var_.items():
    e_ = ext_for(**kw_v)
    mu_, _ = head_mean(view0_, 1, Fn.gelu, e_, tb=lb_)
    d_ = (mu_ - tool_).abs()
    print('%-12s first %.3f  hasl %.3f' % (nm_v, float(d_[m1].mean()), float(d_[m2].mean())))
```