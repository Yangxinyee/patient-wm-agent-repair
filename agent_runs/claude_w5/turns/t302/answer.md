The u = 0 idea is refuted too. Zeroing u at bundles without drugs breaks the original model's match (median difference 4.2) and does not reproduce the tool's edited values. So the rollout feeds the empty-set encoding, as states mode does.

Here is what I know:
- **The difference is specific to the edit:** at hour 13 the tool's edited factual prediction differs from my teacher-forced one (median 2.1 mg/dL), while the original matches within sampling noise. Yet both are supposedly teacher-forced through 12:05.
- **States mode matches exactly:** the tool's edited core equals mine there.

Therefore the dose_edit path must encode some input differently from states mode. It must be an input the original model ignores but my edit is sensitive to. Two kinds of input fit:
- the drug atoms themselves (rate/dose encoding, END atoms);
- tokens that the giant stage-1 values amplify.

Before testing encoding variants, I'll check cheap structural facts in the tool output:
- whether the factual column is shared across the four edits in each model;
- whether the tool-minus-local gap at hour 13 correlates with features of the hour-12 drug bundle: insulin rate, norepinephrine rate, presence of END atoms.

```python
import numpy as np
f13e = d6e[d6e.hour==13].pivot_table(index='stay_id', columns='edit', values='factual')
f13o = d6o[d6o.hour==13].pivot_table(index='stay_id', columns='edit', values='factual')
print('edited factual spread across edits (max-min) median %.4f max %.4f | orig %.4f' % ((f13e.max(1)-f13e.min(1)).median(), (f13e.max(1)-f13e.min(1)).max(), (f13o.max(1)-f13o.min(1)).median()))
feat = []
for sid in RZ.index:
    B = make_bundles(tun[tun.stay_id==sid])
    k = [k for k,(t,ats) in enumerate(B) if 12.0 <= t < 13.0 and any(a.get('kind')==0 and a['code']=='insulin' for a in ats)][0]
    t12, ats = B[k]
    rI = [a['rate'] for a in ats if a.get('kind')==0 and a['code']=='insulin'][0]
    rN = ([a['rate'] for a in ats if a.get('kind')==0 and a['code']=='norepinephrine'] or [0.0])[0]
    nEnd = sum(1 for a in ats if a.get('kind')==2)
    nb_prev = sum(1 for (t,aa) in B[:k] if any(a.get('kind')==0 and a['code'] in ('insulin','norepinephrine') for a in aa))
    raw = tun[(tun.stay_id==sid) & (tun.hour==12) & (tun.event=='insulin')]
    feat.append({'sid': sid, 'rI': rI, 'rN': rN, 'nEnd': nEnd, 'n_drug_bundles_before': nb_prev, 't12': t12, 'k': k, 'raw_rows': len(raw), 'dose_raw': raw.dose.tolist()[:2], 'rate_raw': raw.rate.tolist()[:2]})
FE = pd.DataFrame(feat).set_index('sid').join(RZ[['tool_fact_e','E_m0','tool_fact_o','O_m0']])
FE['gap_e'] = FE.tool_fact_e - FE.E_m0; FE['gap_o'] = FE.tool_fact_o - FE.O_m0
for c in ['rI','rN','nEnd','n_drug_bundles_before']:
    print(c, 'corr with gap_e %.3f, with |gap_e| %.3f' % (np.corrcoef(FE[c], FE.gap_e)[0,1], np.corrcoef(FE[c], FE.gap_e.abs())[0,1]))
print(FE.sort_values('gap_e')[['rI','rN','nEnd','t12','raw_rows','rate_raw','dose_raw','gap_e','gap_o']].round(3).to_string()[:2500])
```