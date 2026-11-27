The error depends on the reading itself. The best stays match to about 0.008, including one whose second bundle is a lone ICU-admission atom with no value. The worst are bundles with high glucose (bins 10–12), and dimension 27 carries most of their error.

So most of the remaining mismatch looks tied to how measurement values are embedded. Three candidates:
- the scale of the z-scored value;
- whether the value-bin embedding is included;
- ties at bin edges, which matter because readings and edges are both integers.

Next I'll vary these: a scale factor on z_value, dropping the bin term, and the ≤ instead of < bin rule. If one variant brings the high-glucose bundles down to the ~0.008 floor, the atom value features are settled.

```python
import copy
def fb_le(code_name, x):
    ed = NUM[(code_name, 'value')]['fine_bin_edges']; return 1 + int(np.sum(np.array(ed) <= x))
inv_code = {v_['id']: v_['name'] for v_ in V['codes']}
def variant_atoms(aa, zscale=1.0, use_bin=True, le=False):
    out_ = []
    for a_ in aa:
        b_ = copy.deepcopy(a_)
        if b_['num'][1] == 1:
            z_ = b_['num'][0] * zscale
            if le:
                nmc = inv_code[b_['code']]; med, iqr = NUM[(nmc, 'value')]['median'], NUM[(nmc, 'value')]['iqr']
                b_['vb'] = fb_le(nmc, a_['num'][0] * iqr + med)
            if not use_bin: b_['vb'] = 0
            b_['num'] = (z_,) + tuple(b_['num'][1:])
        out_.append(b_)
    return out_
for zs, ub, le in [(1, True, False), (1, False, False), (1, True, True), (0.5, True, False), (2, True, False), (0, True, False)]:
    al_ = [variant_atoms(aa, zs, ub, le) for aa in atoms1]
    E_ = torch.stack([setenc_v(aa, 'encode', ACT['gelu'], ACT['relu']) for aa in al_])
    zz_ = mlp2l(torch.cat([ctx_g, E_, U_g], 1), 'z_post', ACT['gelu'])[:, :32]
    er = (zz_ - z1_true).abs().mean(1).numpy()
    print('zscale', zs, 'bin', ub, 'le', le, 'mean err %.4f  worst5 %.4f' % (er.mean(), np.sort(er)[-5:].mean()))
```