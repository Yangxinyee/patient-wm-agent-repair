The reveal worked: single-atom bundles with the same content give identical revealed vectors (spread 1.7e-6), so the values do not depend on the state. My encoder is still far off, though: the best error is 0.40 against a scale of 1.23. Now I have ground truth to debug it piece by piece.

There are two likely culprits:
- A constant offset. If "no actions" is encoded as a fixed vector rather than zero, every measurement bundle's revealed value carries the same added constant.
- The structure of the set encoder: where the size term goes, whether there is a residual, normalisation of the atoms.

Next I will look at whether my error is constant across bundles, and test structural variants on the single-atom bundles. If the error is nearly constant per dimension, it is an offset from the other slot. If a structural variant brings the error near zero, the encoder is identified.

```python
best = res[0]; _, cz, ch, ug, um, ak, ae, aq, avb = best
def atomvec1(cz=0, ch=1, ug=3, um=2, ak=1, ae=1, aq=1, avb=1):
    nf = torch.zeros(len(sub1), 6); nf[:, cz] = zv1t; nf[:, ch] = 1
    ut = 0 if ug is None else P['embed.categorical.unit.weight'][torch.where(isg, torch.tensor(ug), torch.tensor(um))]
    return P['embed.code.weight'][c1] + ak*P['embed.kind.weight'][0] + ae*P['embed.ekind.weight'][1] + aq*P['embed.kq.weight'][0] + ut + avb*P['embed.bins.value.weight'][torch.tensor(vb1)] + lin(nf, 'embed.numeric')
av1 = atomvec1(cz, ch, ug, um, ak, ae, aq, avb)
e_cur = ln(mlp(mlp(av1, 'encode.mlp1', F.gelu) + size1, 'encode.mlp2', F.gelu), 'encode.norm')
dif = (e_cur[:, :32] - r1)
print('diff mean-abs per dim (first 8)', dif.mean(0)[:8].numpy().round(3), 'std per dim', dif.std(0)[:8].numpy().round(3))
# empty-set encodings
u_empty = ln(mlp(lin(torch.zeros(1, 1), 'encode_action.size'), 'encode_action.mlp2', F.gelu), 'encode_action.norm')
print('offset hypothesis: err after adding u_empty', (e_cur[:, :32] + u_empty[:, :32] - r1).abs().mean().item())
# structural variants
size_out = lin(torch.full((len(sub1), 1), np.log(2.0)), 'encode.size')
vars_ = {
 'cur': e_cur,
 'size_outside': ln(mlp(mlp(av1, 'encode.mlp1', F.gelu), 'encode.mlp2', F.gelu) + size_out, 'encode.norm'),
 'no_size': ln(mlp(mlp(av1, 'encode.mlp1', F.gelu), 'encode.mlp2', F.gelu), 'encode.norm'),
 'res1': ln(mlp(av1 + mlp(av1, 'encode.mlp1', F.gelu) + size1, 'encode.mlp2', F.gelu), 'encode.norm'),
 'res2': ln((lambda s: s + mlp(s, 'encode.mlp2', F.gelu))(mlp(av1, 'encode.mlp1', F.gelu) + size1), 'encode.norm'),
 'res12': ln((lambda s: s + mlp(s, 'encode.mlp2', F.gelu))(av1 + mlp(av1, 'encode.mlp1', F.gelu) + size1), 'encode.norm'),
 'ln_atom': ln(mlp(mlp(F.layer_norm(av1, (192,)), 'encode.mlp1', F.gelu) + size1, 'encode.mlp2', F.gelu), 'encode.norm'),
}
for k_, v_ in vars_.items():
    print(k_, round((v_[:, :32] - r1).abs().mean().item(), 4), ' +u_empty', round((v_[:, :32] + u_empty[:, :32] - r1).abs().mean().item(), 4))
```