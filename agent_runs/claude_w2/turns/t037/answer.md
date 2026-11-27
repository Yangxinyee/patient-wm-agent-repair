No combination reproduces the z posterior exactly: the best mean max-error is 0.37. Still, the ranking is consistent: GELU, unit row 2 for map/glucose readings, bins counting edges strictly below the value, `gap(log(dt)/10)`, and the mean in the first 32 outputs. This looks partly right with something missing. Candidates are an extra term in ctx, a different e_k composition (another embedding row or field), or the size feature.

Next I'll diagnose the error with the best combination:
- How large are z values typically, and how well do predicted and actual z correlate across bundles and dimensions?
- Does the error shrink if I vary single components: drop the kq row, drop gap, drop the out term in ctx, or use sum instead of sqrt(n) pooling?

High correlation with a scale or offset gap would mean a small missing piece. Low correlation would mean a structural misunderstanding.

```python
def zpred(ip, ik, a, dt, act='gelu', unit_row=2, side='left', variant=None):
    out = X[ip][:192]; h = X[ip][192:960].reshape(4,192)
    hf = h*torch.exp(-dt*torch.exp(W['core.flow.a']))
    gap = mlp(torch.tensor([np.log(dt)/10],dtype=torch.float64), 'gap', act)
    parts = {'out': out, 'hn': ln(hf.mean(0), 'core.out_norm'), 'gap': gap}
    if variant=='no_out': parts['out'] = 0*out
    if variant=='no_hn': parts['hn'] = 0*out
    if variant=='no_gap': parts['gap'] = 0*out
    if variant=='hn_noflow': parts['hn'] = ln(h.mean(0), 'core.out_norm')
    ctx = parts['out']+parts['hn']+parts['gap']
    x = atom_vec(int(a.code),0,1,a.value,np.nan,np.nan,unit_row,side)
    if variant=='no_kq': x = x - W['embed.kq.weight'][0]
    e = encode([x], act)
    if variant=='e_zero': e = 0*e
    zo = mlp(torch.cat([ctx, e, torch.zeros(192,dtype=torch.float64)]), 'z_post', act)
    return zo[:32]
act_z = X[[c[1] for c in cands], 960:992]
print('actual z: mean abs', act_z.abs().mean().item(), 'sd', act_z.std().item())
for variant in [None,'no_out','no_hn','no_gap','hn_noflow','no_kq','e_zero']:
    P = torch.stack([zpred(ip,ik,a,dt,variant=variant) for (ip,ik,a,dt) in cands])
    err = (P-act_z).abs().mean().item(); corr = np.corrcoef(P.flatten().numpy(), act_z.flatten().numpy())[0,1]
    print(variant, 'mean abs err', round(err,4), 'corr', round(corr,4))
```