Extra categorical rows don't help (all about 0.044), so the remaining end-only error is not in the end atom's embedding. The analogy with e points to the cause. For bundles with no new-record atoms, u may be the action encoder applied to an empty set rather than zero. The e side of action-only bundles worked the same way, even though the documentation says "zero".

I'll test u = empty-set encoding on the end-only bundles, then on all 400 action bundles. If that brings the error to about 0.001, I'll adopt it as the convention.

```python
U_EMPTY = LN(mlp_(lin_(np.array([0.0]), 'encode_action.size'), 'encode_action.mlp2'), P['encode_action.norm.weight'], P['encode_action.norm.bias'])
def err_route3(js, u_empty):
    errs = []
    for j in js:
        t = int(np.floor(hrS[j])); i = sid2i[int(sidS[j])]; Ve, Vu = [], []
        for c, nm, Pn in [(8, 'insulin', ins), (9, 'norepinephrine', ne)]:
            rn, rp = Pn[i, t + 2], Pn[i, t + 1]; df = 4.8 if c == 9 else 1.0
            if not np.isnan(rn) and rn > 0:
                Vu.append(atom_vec(c, kind=0, ekind=2, zd=zsc_(nm, 'dose', rn * df), hd=1.0, zr=zsc_(nm, 'rate', rn), hr=1.0, dbin=fb_(nm, 'dose', rn * df), rbin=fb_(nm, 'rate', rn)))
            if not np.isnan(rp) and rp > 0: Ve.append(atom_vec(c, kind=2, ekind=2))
        e = setenc(np.stack(Ve), 'encode') if Ve else E_EMPTY
        u = setenc(np.stack(Vu), 'encode_action') if Vu else u_empty
        ctx = ctx_from(xs[j - 1], (hrS[j] - hrS[j - 1]) * 3600)
        errs.append(np.abs(xs[j, 960:992] - zmean(ctx, e, u)).mean())
    return np.mean(errs), np.max(errs)
print('END-only, u=empty-enc: %.4f max %.4f' % err_route3(endonly, U_EMPTY))
print('all 400,  u=empty-enc: %.4f max %.4f' % err_route3(jsA, U_EMPTY))
```