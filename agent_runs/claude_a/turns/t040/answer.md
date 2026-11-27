The residual mainly tracks the previous reading: its correlation with the last value is −0.87 for glucose and −0.86 for MAP. The model leans on the last value more than my rebuild does. So the "last" register is still mis-specified: the z_last_value is probably in the wrong column of `last_numeric`, or there is a second copy. The drug-bundle pattern is a smaller, secondary effect.

Next I will speed up the evaluation by holding z at the prior mean and precomputing everything except the register. Then I will search where z_last_value goes among the 8 columns, crossed with the age unit and the position of the has flags. I will also test how sensitive the fit is to the transition term, and regress the reported means on mine plus the last value to measure the gap. If one column placement brings the RMSE down near the 2 mg/dL noise floor, the register is solved.

```python
A = F.gelu
dt_s = dts_opts['s']
hdec = HS[kprev] * torch.exp(-dt_s[:, None, None] * torch.exp(P['core.flow.a']))
ctx0 = OUT[kprev] + ln(hdec.mean(1), 'core.out_norm') + mlp((torch.log(dt_s)/10)[:, None], 'gap', A)
mu0 = mlp(ctx0, 'z_prior', A)[:, :32]
q0 = ln(ctx0 + lin(mu0, 'z_head') + P['type_emb.weight'][1], 'query_norm')
vbt0 = torch.tensor(vb)
base0 = q0 + P['embed.code.weight'][codes_] + P['embed.bins.value.weight'][vbt0] + P['last_numeric.bias']
trans0 = P['embed_transition.value_bin.weight'][codes_*17 + vbt0]
Wln = P['last_numeric.weight']
hz = torch.tensor(has); zz = torch.tensor(zv)
gmT = torch.tensor(gm); mmT = torch.tensor(mm)
def evalf(f, tw=1.0):
    lg = mlp(base0 + f @ Wln.T, 'attr_heads.value_bin', A) + tw*trans0
    m_ = (torch.softmax(lg, -1)[:, 1:] * centM[codes_]).sum(1); e_ = m_ - obs_pm
    return round(e_[gmT].pow(2).mean().sqrt().item(), 3), round(e_[mmT].pow(2).mean().sqrt().item(), 3)
res = []
for age_mult in [1.0, 3600.0]:
    ag = torch.tensor(np.log1p(age_h*age_mult)/10, dtype=torch.float32)
    for c_h, c_a, c_z in itertools.permutations(range(8), 3):
        for c_hv in [None] + [c for c in range(8) if c not in (c_h, c_a, c_z)]:
            f = torch.zeros(len(R), 8); f[:, c_h] = hz; f[:, c_a] = ag; f[:, c_z] = zz
            if c_hv is not None: f[:, c_hv] = hz
            r_ = evalf(f); res.append((r_[0] + r_[1], r_, age_mult, c_h, c_a, c_z, c_hv))
res.sort(key=lambda r: r[0])
for r in res[:10]: print(r)
f = torch.zeros(len(R), 8); f[:, 0] = hz; f[:, 1] = torch.tensor(np.log1p(age_h)/10, dtype=torch.float32); f[:, 2] = zz; f[:, 3] = hz
print('L2 hours trans x0, x1, x2:', evalf(f, 0.0), evalf(f, 1.0), evalf(f, 2.0))
```