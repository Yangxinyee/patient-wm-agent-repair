The END atom is now exact (error 7e-7): code[9] + kind[2] + ekind[2] + kq[0] + numeric.bias, with no unit, bins or numeric values. Every single-reading bundle has n_atoms = 1, so there is no hidden atom.

Given how cleanly the drug and END atoms matched, the error in reading atoms must be in the value part: the bin index rule, the numeric vector, or the z normalisation. Next I'll search over these variants for the single-reading bundles:
- value bin strict, non-strict, shifted down by one, or absent;
- numeric vector [z, 1], [z, 0] or [0, 1];
- z clipped at 10, unclipped, or z·iqr/1.349;
- the unit row (0, 2 or 5);
- with kq[0] and ekind[1] fixed.

One variant reaching error near 1e-6 completes the e_k replica. If none does, the value representation is something else, perhaps a separate numeric scaling for measurements.

```python
def zv_(x, d, mode):
    z = (x - d['median'])/d['iqr']
    return float(np.clip(z, -10, 10)) if mode=='clip' else (float(z) if mode=='raw' else float(np.clip(z*1.349, -10, 10)))
VB = load('embed.bins.value.weight')
resM = []
for ug, um, binv, numv, zm in itertools.product([0,2,5],[0,2,5],['strict','nonstrict','minus1','none'],['z1','z0','01'],['clip','raw','iqr1349']):
    if ug == um and ug != 0: continue
    A = []
    for _, (c_, v_) in singles:
        d = num[(c_,'value')]; ed = np.array(d['fine_bin_edges'])
        b = {'strict': 1+int((ed < v_).sum()), 'nonstrict': 1+int((ed <= v_).sum()), 'minus1': int((ed < v_).sum()), 'none': 0}[binv]
        z = zv_(v_, d, zm)
        nv = {'z1': [z,1.,0,0,0,0], 'z0': [z,0.,0,0,0,0], '01': [0.,1.,0,0,0,0]}[numv]
        a = E_['code'][cid[c_]] + E_['kind'][0] + E_['ekind'][1] + E_['kq'][0] + CAT['unit'][ug if c_.startswith('glucose') else um] + VB[b] + NW @ torch.tensor(nv) + NB
        A.append(a)
    pe = e_from_atoms_batch(torch.stack(A))[:, e_c]
    resM.append((float((pe - TS).abs().max()), ug, um, binv, numv, zm))
resM.sort(key=lambda t: t[0]); print(resM[:4])
```