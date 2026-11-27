The fwd_4 breakdown by hour is decisive about the failure:
- On the factual path (recorded doses), the edited model's rollout drifts upward relative to the original: +2.4 mg/dL at hour 13, rising steadily to +12.1 at hour 24.
- The x2 shift stays near zero at every hour (+0.17 at hour 13).

So in the tool, my edit adds a positive glucose bias at each drug bundle. That bias builds up because in a rollout no real reading resets the belief. Meanwhile the injection hardly responds to the doubled dose. A loop may make it worse: c_hat depends on the believed glucose, so a higher belief raises c_hat, which makes (rate − c_hat) more negative and pushes the belief higher still.

Teacher-forced, my local check showed a slightly better NLL and no bias, which does not fit this picture. Either the tool computes my edit differently from my local copy (fwd_5 showed a ~1e-3 mismatch), or the decoded residual is biased on these stays.

Next I'll use fwd_5, the factual states of both models on 13 tuning stays, to compare the tool's edited-minus-original state differences with my local ones. I'll locate the recorded inputs (`recs`, `run_cmp`) that I apparently built for this.

- If the differences match, the edit itself is biased and blind to doses even in my exact local model, and the design must change.
- If they don't match, the tool's computation differs from mine.

```python
import numpy as np
for nm in ['recs','recs2','stays_t','cases','caseinfo','tests','res_t','res_tc','res_tf']:
    v = globals().get(nm)
    if v is None: continue
    info = type(v).__name__
    if isinstance(v,(list,tuple)): info += f' len {len(v)} el0 {type(v[0]).__name__}' + (f' keys {list(v[0].keys())[:8]}' if isinstance(v[0],dict) else '')
    elif isinstance(v,dict): info += f' keys {list(v.keys())[:8]}'
    print(nm, info)
z5o = np.load('/work/fwd_5/states_original.npz'); z5e = np.load('/work/fwd_5/states_edited.npz')
Xo, Xe = z5o['x'].astype(np.float64), z5e['x'].astype(np.float64)
hasI, hasN = z5o['has_insulin'].astype(bool), z5o['has_norepinephrine'].astype(bool)
d = Xe - Xo
def rn(a,b): return np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1)
for nm_, sl in [('out',slice(0,192)),('h',slice(192,960)),('z',slice(960,992))]:
    r_ = rn(d[:,sl], Xo[:,sl])
    print(nm_, 'rel diff: ins rows %.2e  ne rows %.2e  other %.2e' % (np.median(r_[hasI]), np.median(r_[hasN & ~hasI]), np.median(r_[~hasI & ~hasN])))
print('n rows', len(Xo), 'ins', hasI.sum(), 'ne', hasN.sum())
```