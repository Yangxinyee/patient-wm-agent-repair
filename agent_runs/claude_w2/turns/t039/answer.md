The identity holds to float32 precision (max error 1e-6): out = core.out_norm(mean of the h slots). The state layout and out_norm are right. The h slots carry large values (norms around 600-800), which is worth remembering for any edit to h.

So the z_post mismatch comes from ctx/gap or from e_k (atom embedding or encoder). The architecture's embedding list does not mention `embed.kq` or `embed.text`, and dropping kq helped slightly. I'll redo the sweep without kq, varying:
- the unit row (0-5) and the bin side;
- the gap input: dt in seconds, minutes or hours, or no gap;
- the GELU variant.

An exact match (error ~1e-6) settles the conventions for measurement atoms. A persistent ~0.1 error means the problem is elsewhere, for example in the encoder pooling or the z_post input order.

```python
def atom_vec2(code, kind, ekind, value, rate, dose, unit_row, side, rate_unit_row=0, use_kq=False):
    x = W['embed.code.weight'][code] + W['embed.kind.weight'][kind] + W['embed.ekind.weight'][ekind] + W['embed.categorical.unit.weight'][unit_row]
    if rate_unit_row: x = x + W['embed.categorical.rate_unit.weight'][rate_unit_row]
    if use_kq: x = x + W['embed.kq.weight'][0]
    num = torch.zeros(6, dtype=torch.float64); nm = names[code]
    if not np.isnan(value):
        num[0] = zsc(nm,'value',value); num[1] = 1; x = x + W['embed.bins.value.weight'][fbin(nm,'value',value,side)]
    if not np.isnan(dose):
        num[2] = zsc(nm,'dose',dose); num[3] = 1; x = x + W['embed.bins.dose.weight'][fbin(nm,'dose',dose,side)]
    if not np.isnan(rate):
        num[4] = zsc(nm,'rate',rate); num[5] = 1; x = x + W['embed.bins.rate.weight'][fbin(nm,'rate',rate,side)]
    return x + W['embed.numeric.weight'] @ num + W['embed.numeric.bias']
def ctx_v(ip, dt, gscale):
    out = X[ip][:192]; h = X[ip][192:960].reshape(4,192)
    hf = h*torch.exp(-dt*torch.exp(W['core.flow.a']))
    c = out + ln(hf.mean(0), 'core.out_norm')
    if gscale is not None:
        c = c + mlp(torch.tensor([np.log(dt/gscale)/10],dtype=torch.float64), 'gap', 'gelu')
    return c
res=[]
for unit_row, side, gscale in itertools.product(range(6), ['left','right'], [1.0, 60.0, 3600.0, None]):
    P=[]
    for (ip,ik,a,dt) in cands:
        e = encode([atom_vec2(int(a.code),0,1,a.value,np.nan,np.nan,unit_row,side)], 'gelu')
        zo = mlp(torch.cat([ctx_v(ip,dt,gscale), e, torch.zeros(192,dtype=torch.float64)]), 'z_post', 'gelu')
        P.append(zo[:32])
    P = torch.stack(P); err = (P-act_z).abs().mean().item()
    res.append((round(err,5), unit_row, side, gscale))
res.sort(); print(res[:10])
```