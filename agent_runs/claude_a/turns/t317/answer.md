The last pass ran the factual check on 100 tuning stays. On those stays the edited model fits slightly better than the original: value NLL 1.6485 against 1.6501, so −0.0016 nats. Glucose_finger improved, MAP_arterial was unchanged (1.5331 against 1.5330), glucose_lab was +0.003 and map_cuff −0.003. This is well inside the 0.005 limit and fits the offline estimate of about +0.0005 on all 400 tuning stays, given the sampling noise of 100 stays.

My notes say passes 11 to 14 were meant to be the factual check of the final edit on all 400 tuning stays, in four batches. Before giving the final answer I will pool those four batches in Python. I will check that they cover different stays, measure the pooled cost and its standard error by event, and bootstrap the chance that a random set of 100, 200 or 400 stays would cost more than 0.005. I will also list the kernel functions, in case I need to back off the norepinephrine readout, and write the result to the notebook.

How I will read the result:
- **Confirms:** a pooled cost near or below about +0.0005 with a small tail probability. I then submit as is.
- **Refutes:** a pooled cost above about 0.003, or a heavy tail. I would then shrink the norepinephrine half-dose term (thh) or remove it.

I am deliberately not raising thh to fix the low norepinephrine x1/2 readout. The cost grows as about 0.0028·thh², and the gain would be only about 0.1 to 0.3 of the 0.95 gap.

```python
import os, json, re, inspect
import numpy as np, pandas as pd
log = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.append(s)

# 1. what are fwd_6..14
for k in range(6, 15):
    d = f'/work/fwd_{k}'
    if not os.path.isdir(d): continue
    files = sorted(os.listdir(d))
    P(f'fwd_{k}:', files)
    for f in files:
        if f.endswith('.json'):
            try:
                js = json.load(open(os.path.join(d, f)))
                P('   ', f, json.dumps(js)[:300])
            except Exception as e:
                P('   json err', f, e)

summ = []
# 2. factual pooling
try:
    fact = {}
    for k in range(6, 15):
        d = f'/work/fwd_{k}'
        fo, fe = os.path.join(d, 'factual_original.csv'), os.path.join(d, 'factual_edited.csv')
        if os.path.exists(fo) and os.path.exists(fe):
            fact[k] = (pd.read_csv(fo), pd.read_csv(fe))
    P('factual batches:', list(fact))
    a, b = fact[max(fact)]
    P('cols', list(a.columns)); P(a.head(3).to_string())
    num = a.select_dtypes('number'); numb = b.select_dtypes('number')
    P('means last orig:', {c: round(float(num[c].mean()), 5) for c in num.columns})
    P('means last edit:', {c: round(float(numb[c].mean()), 5) for c in numb.columns})
    target = 1.65009
    cands = [c for c in num.columns if 'nll' in c.lower()] or list(num.columns)
    ncol = min(cands, key=lambda c: abs(float(num[c].mean()) - target))
    scol = [c for c in a.columns if 'stay' in c.lower()][0]
    ecs = [c for c in a.columns if c.lower() in ('event','event_type','code','name','kind','type')]
    ecol = ecs[0] if ecs else None
    P('nll col', ncol, 'stay col', scol, 'event col', ecol)
    rows = []
    for k,(a_,b_) in sorted(fact.items()):
        assert len(a_)==len(b_)
        dd = b_[ncol].values - a_[ncol].values
        st = a_[scol].values
        P(f'fwd_{k}: nst={len(np.unique(st))} n={len(a_)} orig={a_[ncol].mean():.5f} ed={b_[ncol].mean():.5f} diff={dd.mean():+.5f}')
        rows.append(pd.DataFrame({'k':k,'stay':st,'d':dd,'ev':(a_[ecol].values if ecol else 'all')}))
    R = pd.concat(rows)
    Rf = R[R.k>=11]
    P('k>=11: rows', len(Rf), 'unique stays', Rf.stay.nunique(), 'stays per k', Rf.groupby('k').stay.nunique().to_dict())
    Rf = Rf.drop_duplicates(subset=None) if False else Rf
    g = Rf.groupby('stay').d.agg(['sum','count'])
    m = g['sum'].sum()/g['count'].sum()
    se = np.sqrt(((g['sum'] - m*g['count'])**2).sum())/g['count'].sum()
    P('POOLED diff', round(m,5), 'se', round(se,5))
    bev = Rf.groupby('ev').d.agg(['mean','count']).round(5)
    P('by event', bev.to_dict())
    rng = np.random.default_rng(0)
    S = g['sum'].values; C = g['count'].values
    tail = {}
    for N in (100, 200, 400):
        idx = rng.integers(0, len(S), size=(5000, N))
        v = S[idx].sum(1)/C[idx].sum(1)
        tail[N] = (round(float(np.percentile(v,99)),5), float((v>0.005).mean()))
    P('bootstrap (p99, P>.005):', tail)
    summ.append(f'pooled {Rf.stay.nunique()} stays diff {m:+.5f} (se {se:.5f}); by event {bev["mean"].to_dict()}; bootstrap {tail}')
except Exception as e:
    import traceback; P('factual err', repr(e)); traceback.print_exc()

# 3. available functions
try:
    fn = sorted(n for n,v in list(globals().items()) if inspect.isfunction(v) and not n.startswith('_'))
    P('kernel fns:', fn)
    for f in ['lib1.py','lib2.py','lib3.py','lib4.py']:
        try:
            src = open('/work/'+f).read()
            P(f, re.findall(r'^def (\w+)\(', src, flags=re.M))
        except Exception as e:
            P(f, 'err', e)
except Exception as e:
    P('fn err', e)

open('/work/run294.txt','w').write('\n'.join(log))
with open('/work/notes.md','a') as fh:
    fh.write('\n## Restart 9 (run 294-)\n- fwd14 factual (100 tuning, edit hash 5c2b8ab571 = th2 .868 thh .5 f 1.16): orig 1.65009 ed 1.6485 (-0.0016); MAP_art +.0001, gluc_f -.003.\n'
             '- Run294 pooled factual fwd11-14: ' + (summ[0] if summ else 'FAILED (see /work/run294.txt)') + '\n'
             '- Decision: do NOT push thh (NE half) further: cost ~.0028*thh^2, gain on ne_half small; finalize unless pooled cost bad.\n')
print('notes appended')
```