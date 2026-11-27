These magnitudes explain a lot. The stage-1 kv edit gives huge value vectors to every token except u: about 1e3–5e3 for the slots, up to 4e3 for e, and 4e3–2e4 for the z token. Only u's own change is small: 5–17 at drug bundles and exactly 0 for the empty action, as designed. The edit therefore depends on head 1 attending almost entirely to u, plus some cancellation inside W_o.

Raw float32 rounding of the value product is only about 1e-3, which is probably not the whole story. Before spending a forward pass on a states check, there is a cheaper candidate for the hour-13 asymmetry: the tool may scale the drug atoms differently from my local simulation, for example the rate only rather than rate and dose. My decoder was fitted with both scaled.

I'll do two things:
- grep the documentation for exactly what the dose edit changes;
- compute the local single-bundle response under three variants: rate and dose scaled together, rate only, and dose only.

If one variant gives x2 ≈ −3 and x1/2 ≈ 0 at the bundle level, as the tool does at hour 13, that is the mechanism.

```python
import re, numpy as np, time
for fn_doc in ['/model/ARCHITECTURE.md', '/data/README.md']:
    L_ = open(fn_doc).read().splitlines()
    for i, l in enumerate(L_):
        if re.search(r'dose|scal|edit|rollout|hidden|draw', l, re.I): print(fn_doc.split('/')[-1], i, l[:220])
def scale_var(ats, drug, f, mode):
    out = []
    for a in ats:
        a2 = dict(a)
        if a.get('kind')==0 and a['code']==drug:
            if mode in ('both','rate') and 'rate' in a2: a2['rate'] = a2['rate']*f
            if mode in ('both','dose') and 'dose' in a2: a2['dose'] = a2['dose']*f
        out.append(a2)
    return out
rng_ = np.random.default_rng(0); recs_v = []; t0 = time.time()
for sid in rng_.choice(SIDS, 300, replace=False):
    Bs, Ss = bundles_B[sid], steps_all[sid]
    for j,(t,ats) in enumerate(Bs):
        k0 = [a for a in ats if a.get('kind')==0 and a['code'] in ('insulin','norepinephrine')]
        for drug in sorted(set(a['code'] for a in k0)):
            rate = [a['rate'] for a in k0 if a['code']==drug][0]; X_ = 'I' if drug=='insulin' else 'N'
            R0 = internals(WE, Ss[j]); x0 = float((R0['n3'][:4]@EP_['wx'+X_]+EP_['bx'+X_]).mean())
            rec = {'drug': drug, 'rate': rate}
            for mode in ['both','rate','dose']:
                for f in [2.0, 0.5]:
                    e2,u2 = enc2(scale_var(ats, drug, f, mode)); d2 = dict(Ss[j]); d2['u'] = u2
                    R = internals(WE, d2); rec[f'{mode}{f}'] = float((R['n3'][:4]@EP_['wx'+X_]+EP_['bx'+X_]).mean()) - x0
            recs_v.append(rec)
print('n', len(recs_v), 'time %.0f' % (time.time()-t0))
DV = pd.DataFrame(recs_v)
for drug, g in DV.groupby('drug'):
    print(drug, ' '.join(f'{c}: {((g[c])*(g.rate)).sum()/(g.rate**2).sum():.3f}' for c in ['both2.0','both0.5','rate2.0','rate0.5','dose2.0','dose0.5']), '(slope of dx on rate)')
```