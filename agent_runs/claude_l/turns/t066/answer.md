The value-feature variants change nothing useful. A z-scale of 1 is best, and dropping the bin term or changing the tie rule barely matters. So the residual error of about 0.04 is not in how values are encoded.

One untested convention is the unit-table index mapping. I inferred mg/dL→2 and mmHg→5 earlier, but both are high-frequency units with similarly trained rows, so they could be swapped. Z dimension 27 carries the high-glucose errors, which fits a unit problem specific to glucose.

I'll test the mg/dL/mmHg swap on the measurement bundles. If the error drops sharply, the mapping was wrong; if not, I'll move on to testing a full state update.

```python
def swap_units(aa, mp):
    out_ = []
    for a_ in aa:
        b_ = copy.deepcopy(a_); b_['unit'] = mp.get(b_['unit'], b_['unit']); out_.append(b_)
    return out_
for nm_, mp in [('orig', {}), ('swap mgdl<->mmHg', {2: 5, 5: 2}), ('mgdl->1', {2: 1}), ('mmHg->1', {5: 1}), ('both->1', {2: 1, 5: 1}), ('mgdl->3', {2: 3}), ('mmHg->4', {5: 4})]:
    al_ = [swap_units(aa, mp) for aa in atoms1]
    E_ = torch.stack([setenc_v(aa, 'encode', ACT['gelu'], ACT['relu']) for aa in al_])
    zz_ = mlp2l(torch.cat([ctx_g, E_, U_g], 1), 'z_post', ACT['gelu'])[:, :32]
    er = (zz_ - z1_true).abs().mean(1).numpy(); print(nm_, 'mean err %.4f worst5 %.4f' % (er.mean(), np.sort(er)[-5:].mean()))
```