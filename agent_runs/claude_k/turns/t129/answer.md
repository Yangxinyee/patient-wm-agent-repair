Every bundle containing only new-record atoms is exact (0.0007–0.002), for both drugs and their mixtures. The end-of-infusion atoms are wrong under both conventions: end-only bundles miss by about 0.19. So the end atom's embedding differs in some categorical term or attribute flag I haven't guessed.

I'll search over variants on the 19 bundles containing only end atoms:
- atom kind 0, 1 or 2;
- attributes none, rate only, dose only, or both;
- one optional extra categorical row from the small tables (action, status, route, rate unit, unit, kq, text).

A variant reaching about 0.001 identifies the end atom. Then u_k is exact for every action bundle, including under edited rates.

```python
CATS = {t: P[f'embed.categorical.{t}.weight'] for t in ('action', 'status', 'route', 'rate_unit', 'unit')}
CATS['kq'] = P['embed.kq.weight']; CATS['text'] = P['embed.text.weight']
extras = [('none', None)] + [(f'{t}{r}', CATS[t][r]) for t in CATS for r in range(CATS[t].shape[0])]
endonly = [j for j in jsA if comp(sid2i[int(sidS[j])], int(np.floor(hrS[j])))[0::2] == (0, 0)]
def end_atom(c, nm, r, kind, att, ex):
    dfac = 4.8 if c == 9 else 1.0
    kw = dict(kind=kind, ekind=2)
    if att in ('rate', 'both'): kw.update(zr=zsc_(nm, 'rate', r), hr=1.0, rbin=fb_(nm, 'rate', r))
    if att in ('dose', 'both'): kw.update(zd=zsc_(nm, 'dose', r * dfac), hd=1.0, dbin=fb_(nm, 'dose', r * dfac))
    v = atom_vec(c, **kw); return v if ex is None else v + ex
def err_end(kind, att, ex):
    errs = []
    for j in endonly:
        t = int(np.floor(hrS[j])); i = sid2i[int(sidS[j])]; V = []
        for c, nm, Pn in [(8, 'insulin', ins), (9, 'norepinephrine', ne)]:
            rp = Pn[i, t + 1]
            if not np.isnan(rp) and rp > 0: V.append(end_atom(c, nm, rp, kind, att, ex))
        ctx = ctx_from(xs[j - 1], (hrS[j] - hrS[j - 1]) * 3600)
        errs.append(np.abs(xs[j, 960:992] - zmean(ctx, E_EMPTY, setenc(np.stack(V), 'encode_action'))).mean())
    return np.mean(errs)
out = sorted((err_end(k, a, ex), k, a, en) for k in (0, 1, 2) for a in ('none', 'rate', 'dose', 'both') for en, ex in extras)
print('n end-only', len(endonly)); print([(round(e, 4), k, a, en) for e, k, a, en in out[:6]])
```