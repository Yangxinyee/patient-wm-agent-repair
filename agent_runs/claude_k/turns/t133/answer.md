Using the empty-set encoding for u brings end-only bundles to 0.0042 and all 400 action bundles to 0.0035 (max 0.022). That is close, but not yet the 0.0006 precision of new-record bundles.

The remaining error sits in bundles with end atoms. Under the corrected routing, those atoms may still carry the ending record's rate and/or dose. I'll retest the four attribute variants on the end-only bundles.

```python
def err_route4(js, att):
    errs = []
    for j in js:
        t = int(np.floor(hrS[j])); i = sid2i[int(sidS[j])]; Ve, Vu = [], []
        for c, nm, Pn in [(8, 'insulin', ins), (9, 'norepinephrine', ne)]:
            rn, rp = Pn[i, t + 2], Pn[i, t + 1]; df = 4.8 if c == 9 else 1.0
            if not np.isnan(rn) and rn > 0:
                Vu.append(atom_vec(c, kind=0, ekind=2, zd=zsc_(nm, 'dose', rn * df), hd=1.0, zr=zsc_(nm, 'rate', rn), hr=1.0, dbin=fb_(nm, 'dose', rn * df), rbin=fb_(nm, 'rate', rn)))
            if not np.isnan(rp) and rp > 0: Ve.append(end_atom(c, nm, rp, 2, att, None))
        e = setenc(np.stack(Ve), 'encode') if Ve else E_EMPTY
        u = setenc(np.stack(Vu), 'encode_action') if Vu else U_EMPTY
        ctx = ctx_from(xs[j - 1], (hrS[j] - hrS[j - 1]) * 3600)
        errs.append(np.abs(xs[j, 960:992] - zmean(ctx, e, u)).mean())
    return np.mean(errs), np.max(errs)
for att in ('none', 'rate', 'dose', 'both'):
    print(att, 'END-only %.4f/%.4f' % err_route4(endonly, att), '| all %.4f/%.4f' % err_route4(jsA, att))
```