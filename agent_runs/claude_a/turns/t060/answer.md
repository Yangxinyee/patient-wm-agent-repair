No single unit row makes glucose_finger exact: row 3 gives 0.046 and row 4 gives 0.049. MAP and glucose_lab are exact, so the remaining error for glucose_finger is probably in how its number is scored or binned, not in the unit.

Next I will look at how the glucose_finger error varies with the value, and test alternatives:
- glucose_lab's normalisation or bin edges;
- no clipping;
- adding the unused unit row 1 or the other categorical rows.

If the error grows with the value and vanishes under another normalisation, the z-score is the problem. If it is constant, a missing embedding term is.

```python
gm1 = (sub1.event == 'glucose_finger').values
subg = sub1[gm1]; rg = r1[gm1]; cg = c1[gm1]
def e_gf(zvals, vbins, extra=0, unit=3):
    nf = torch.zeros(len(subg), 6); nf[:, 0] = torch.tensor(zvals, dtype=torch.float32); nf[:, 1] = 1
    av_ = P['embed.code.weight'][cg] + P['embed.kind.weight'][0] + P['embed.ekind.weight'][1] + P['embed.kq.weight'][0] + P['embed.categorical.unit.weight'][unit] + P['embed.bins.value.weight'][torch.tensor(vbins)] + lin(nf, 'embed.numeric') + extra
    e_ = ln(mlp(mlp(av_, 'encode.mlp1', F.gelu) + lin(torch.full((len(subg), 1), np.log(2.0)), 'encode.size'), 'encode.mlp2', F.gelu), 'encode.norm')
    return (e_[:, :32] + u_empty[:, :32] - rg).abs().mean(1)
vg = subg.value.values
z_std = zsc('glucose_finger', 'value', vg); b_std = fbin('glucose_finger', 'value', vg)
z_lab = zsc('glucose_lab', 'value', vg); b_lab = fbin('glucose_lab', 'value', vg)
er_ = e_gf(z_std, b_std)
print(pd.DataFrame({'v': vg, 'err': er_.numpy()}).sort_values('v').iloc[::5].round(4).T.to_string())
for nm_, zz_, bb_ in [('std', z_std, b_std), ('lab z', z_lab, b_std), ('lab bins', z_std, b_lab), ('lab both', z_lab, b_lab)]:
    for u_ in [3, 4]:
        print(nm_, 'unit', u_, round(e_gf(zz_, bb_, unit=u_).mean().item(), 5))
for nm_, ex_ in [('unit1', P['embed.categorical.unit.weight'][1]), ('status1', P['embed.categorical.status.weight'][1]), ('route1', P['embed.categorical.route.weight'][1]), ('text1', P['embed.text.weight'][1]), ('kq1', P['embed.kq.weight'][1] - P['embed.kq.weight'][0]), ('kq2', P['embed.kq.weight'][2] - P['embed.kq.weight'][0]), ('kq3', P['embed.kq.weight'][3] - P['embed.kq.weight'][0])]:
    print('extra', nm_, round(e_gf(z_std, b_std, extra=ex_).mean().item(), 5))
```