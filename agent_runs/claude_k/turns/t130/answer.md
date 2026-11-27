No embedding variant fixes the end-only bundles; the best still misses by 0.128. So the end atoms are probably not wrongly embedded but routed differently. They may not count as actions at all. They could go through the measurement encoder into e_k, leaving u_k as the new-record atoms only, or zero when there are none.

I'll test that routing on the 19 end-only bundles, with the end atom bare or carrying its attributes. If one variant gets to about 0.001, I'll confirm it on all 400 action bundles.

```python
def err_route(att, ex_name='none', js=endonly, starts=True):
    ex = dict(extras)[ex_name]; errs = []
    for j in js:
        t = int(np.floor(hrS[j])); i = sid2i[int(sidS[j])]; Ve, Vu = [], []
        for c, nm, Pn in [(8, 'insulin', ins), (9, 'norepinephrine', ne)]:
            rn, rp = Pn[i, t + 2], Pn[i, t + 1]
            if starts and not np.isnan(rn) and rn > 0:
                df = 4.8 if c == 9 else 1.0
                Vu.append(atom_vec(c, kind=0, ekind=2, zd=zsc_(nm, 'dose', rn * df), hd=1.0, zr=zsc_(nm, 'rate', rn), hr=1.0, dbin=fb_(nm, 'dose', rn * df), rbin=fb_(nm, 'rate', rn)))
            if not np.isnan(rp) and rp > 0: Ve.append(end_atom(c, nm, rp, 2, att, ex))
        e = setenc(np.stack(Ve), 'encode') if Ve else E_EMPTY
        u = setenc(np.stack(Vu), 'encode_action') if Vu else np.zeros(192)
        ctx = ctx_from(xs[j - 1], (hrS[j] - hrS[j - 1]) * 3600)
        errs.append(np.abs(xs[j, 960:992] - zmean(ctx, e, u)).mean())
    return np.mean(errs)
for att in ('none', 'rate', 'dose', 'both'):
    print('END via encode, attrs', att, '%.4f' % err_route(att))
print('all 400, attrs both: %.4f | none: %.4f' % (err_route('both', js=jsA), err_route('none', js=jsA)))
```