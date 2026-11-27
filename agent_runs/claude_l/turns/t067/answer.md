Swapping the units was the fix: with mg/dL→5 and mmHg→2, the z error drops from 0.040 to 0.011. My earlier mapping had them reversed. Since the earlier activation choices were made under the wrong mapping, I'll redo the search with the corrected units.

The search covers:
- the encoder activations and pooling;
- whether `kq` is used;
- the activation in the empty-action encoding;
- the `gap` and `z_post` activations.

If the error falls to about 1e-5 for some combination, the measurement encoder and `z_post` are settled. The drug-unit mappings (units vs mg, and units/hour vs mcg/kg/min) may need the same swap check, which I will make on drug bundles.

```python
atoms1u = [swap_units(aa, {2: 5, 5: 2}) for aa in atoms1]
rs = []
for a1n in ['gelu', 'relu', 'silu', 'gelu_t']:
    for a2n in ['gelu', 'relu', 'silu', 'gelu_t']:
        for kq in [True, False]:
            for pool in ['sqrt', 'sum']:
                E_ = torch.stack([setenc_v(aa, 'encode', ACT[a1n], ACT[a2n], False, False, pool, kq) for aa in atoms1u])
                for ua in ['gelu', 'relu', 'silu', 'zero']:
                    U_ = torch.zeros_like(E_) if ua == 'zero' else empty_enc('encode_action', ACT[ua])[None].expand_as(E_)
                    for gan in ['gelu', 'silu', 'relu']:
                        ctx_ = variants['full_flow'] + mlp2l(torch.tensor(np.log(dts)/10)[:, None], 'gap', ACT[gan])
                        for zn in ['gelu', 'relu', 'silu', 'gelu_t']:
                            zz_ = mlp2l(torch.cat([ctx_, E_, U_], 1), 'z_post', ACT[zn])[:, :32]
                            er = (zz_ - z1_true).abs(); rs.append((er.mean().item(), er.max().item(), a1n, a2n, kq, pool, ua, gan, zn))
rs.sort(key=lambda r: r[0])
for r_ in rs[:10]: print(np.round(r_[:2], 6), r_[2:])
```