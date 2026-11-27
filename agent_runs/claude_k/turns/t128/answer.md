Atom counts match everywhere, but neither convention for end-of-infusion atoms is exact over all 400 action bundles (mean error 0.055–0.068). Single new-record action-A bundles were exact, so the error must come from end atoms, action-B atoms, or both. Two places I may be wrong:
- I assumed action-B dose = rate × 4.8, and the records may say otherwise;
- end atoms may carry only some attributes.

I'll check the dose/rate ratio in the records. Then I'll split the z error by bundle composition (which of the four atom types are present) under the two end conventions. That will show which atom type needs fixing.

```python
nr = tr[tr.event == 'norepinephrine']; ir = tr[tr.event == 'insulin']
print('ne dose/rate ratio', np.round(np.nanpercentile(nr.dose / nr.rate, [1, 50, 99]), 3), '| ins', np.round(np.nanpercentile(ir.dose / ir.rate, [1, 50, 99]), 3))
def comp(i, t):
    f = lambda Pn, c: (0 <= c < Pn.shape[1]) and (not np.isnan(Pn[i, c])) and Pn[i, c] > 0
    return (int(f(ins, t + 2)), int(f(ins, t + 1)), int(f(ne, t + 2)), int(f(ne, t + 1)))
from collections import defaultdict
res = {em: defaultdict(list) for em in ('attrs', 'bare')}
for j in jsA:
    t = int(np.floor(hrS[j])); i = sid2i[int(sidS[j])]; cp = comp(i, t)
    ctx = ctx_from(xs[j - 1], (hrS[j] - hrS[j - 1]) * 3600)
    for em in res:
        V = drug_atoms(i, t, em); res[em][cp].append(np.abs(xs[j, 960:992] - zmean(ctx, E_EMPTY, setenc(np.stack(V), 'encode_action'))).mean())
for cp in sorted(res['attrs']):
    print('iS iE nS nE', cp, 'n=%d' % len(res['attrs'][cp]), 'attrs %.4f' % np.mean(res['attrs'][cp]), 'bare %.4f' % np.mean(res['bare'][cp]))
```