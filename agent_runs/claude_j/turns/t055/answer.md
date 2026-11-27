No variant matches. Exact GELU with gap input log(dt in seconds)/10 is best but still off by 0.28, and the tanh-GELU is indistinguishable. So the activation and the gap convention are probably right, and the error is elsewhere.

To find which input is wrong, I'll check how much each one matters. I'll compare the error against the typical size of z when the measurement vector is zeroed, the gap term dropped, the out_norm part of the context dropped, or h used without time-flow decay. The ablation with the least effect on the error points to the faulty component. I'll also test a few embedding variants: no kq term, `<=` instead of `<` for bin edges, other kq rows.

```python
ACT = F.gelu
print('z std over bundles', Xs[:,960:992].std(0).mean().item(), 'z abs mean', Xs[:,960:992].abs().mean().item())
def z_err3(ntest=30, e_mode='full', ctx_mode='full', kq=0, le=False, has_val=1.0):
    errs=[]
    for r_, rp, dt, ms_ in tests[:ntest]:
        atoms=[]
        for c, v in ms_:
            x = Wt['embed.code.weight'][code_id[c]] + Wt['embed.kind.weight'][0] + Wt['embed.ekind.weight'][1]
            if kq is not None: x = x + Wt['embed.kq.weight'][kq]
            x = x + Wt['embed.categorical.unit.weight'][UNIT['mmHg' if c.startswith('map') else 'mg/dL']]
            ed = np.array(NUM[(c,'value')]['fine_bin_edges']); b_ = int(1 + ((ed <= v) if le else (ed < v)).sum())
            x = x + Wt['embed.bins.value.weight'][b_]
            x = x + Wt['embed.numeric.weight'] @ torch.tensor([zsc(c,'value',v), has_val, 0., 0., 0., 0.]) + Wt['embed.numeric.bias']
            atoms.append(x)
        e_ = encode(atoms) if e_mode=='full' else torch.zeros(192)
        out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
        dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        t1 = ln((h_p*dec).mean(0), 'core.out_norm'); t2 = mlp(torch.tensor([math.log(dt)/10]), 'gap')
        ctx = {'full': out_p+t1+t2, 'nogap': out_p+t1, 'nonorm': out_p+t2, 'nodecay': out_p+ln(h_p.mean(0),'core.out_norm')+t2, 'zero': torch.zeros(192)}[ctx_mode]
        zp = mlp(torch.cat([ctx, e_, torch.zeros(192)]), 'z_post')[:32]
        errs.append((zp - Xs[r_, 960:992]).abs().max().item())
    return round(float(np.median(errs)),4)
for em in ['full','zero']:
    for cm in ['full','nogap','nonorm','nodecay','zero']:
        print(em, cm, z_err3(30, em, cm))
for kq_ in [None,0,1,2,3]:
    print('kq', kq_, z_err3(30, kq=kq_), 'le', z_err3(30, kq=kq_, le=True))
```