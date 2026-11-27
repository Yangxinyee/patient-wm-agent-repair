Nothing ran in the last step. The output was withheld, and the counts still read 15 forward passes and 297 Python runs. The current saved edit is v10b. It is the version meant to remove the leak before the first drug bundle, which was behind the earlier factual-fit failures. v10b has not been tested yet.

I have only 3 Python runs left, so I will use one now to recover what I need to test v10b with forward passes, which cost no Python runs:
- the stay ids for the requests,
- my earlier estimates of the true drug effects from the notebook,
- the variable names needed to rebuild a weaker version if necessary.

I will also back up v10b to a file and log the plan.

Next I will run v10b through factual "both" passes on the tuning stays, and a dose_edit "both" pass on the 100 selected stays.

- **Confirmed:** the value-NLL change is at most about +0.003 on 400 stays, and the dose shifts move toward my truth estimates.
- **Refuted:** the NLL change is still large, or appears in stays without drugs. That would mean the leak fix did not hold, and I would rebuild with lower gains or fall back to the original weights.

```python
import json, numpy as np, pandas as pd, inspect, torch
def cv(x):
    try:
        return int(x)
    except Exception:
        return str(x)
def rng(a):
    try:
        a = sorted(int(x) for x in a)
    except Exception:
        return None
    r = []; s0 = p = a[0]
    for x in a[1:]:
        if x == p + 1:
            p = x
            continue
        r.append(f"{s0}-{p}" if p > s0 else f"{s0}")
        s0 = p = x
    r.append(f"{s0}-{p}" if p > s0 else f"{s0}")
    return r
tids = []
try:
    hdr = open('/data/tuning.csv').readline().strip().split(',')
    sc = [c for c in hdr if 'stay' in c.lower()][0]
    tu = pd.read_csv('/data/tuning.csv', usecols=[sc])[sc]
    tids = [cv(x) for x in pd.unique(tu)]
    rr = rng(tids)
    print('TUNE col', sc, 'n', len(tids), 'nranges', len(rr) if rr else None, (rr if rr and len(rr) <= 8 else ''))
    if not rr or len(rr) > 8:
        print('TUNE200', json.dumps(tids[:200], separators=(',', ':')))
except Exception as ex:
    print('tune err', repr(ex)[:150])
try:
    s = json.load(open('/work/sel100_both.json'))
    ids = s.get('stays', s) if isinstance(s, dict) else s
    if isinstance(s, dict):
        print('sel meta', str({k: v for k, v in s.items() if k != 'stays'})[:300])
    ids = [cv(x) for x in ids]
    st = set(tids)
    print('SEL n', len(ids), 'in tuning', sum(x in st for x in ids))
    print('SEL100', json.dumps(ids, separators=(',', ':')))
except Exception as ex:
    print('sel err', repr(ex)[:150])
try:
    L = open('/work/notes.md').read().splitlines()
    print('notes lines', len(L))
    for i, l in enumerate(L[:12]):
        print(i, l[:140])
    keys = ('truth', 'TRUTH', 'true eff', 'True eff', 'TRUE', 'target', 'belie', 'BELIE', 'expect', 'readout')
    hits = [(i, l) for i, l in enumerate(L) if i >= 12 and any(k in l for k in keys)]
    print('hits', len(hits))
    for i, l in (hits if len(hits) <= 26 else hits[:12] + hits[-14:]):
        print(i, l[:140])
except Exception as ex:
    print('notes err', repr(ex)[:150])
try:
    nm = []
    for k, v in list(globals().items()):
        if not k.startswith('_') and isinstance(v, (np.ndarray, torch.Tensor)) and k[0] in 'Bt' and len(k) <= 8:
            nm.append(f"{k}{tuple(v.shape)}")
    print('vars', ' '.join(nm)[:800])
    print('build7', inspect.signature(build7))
    for k in ('tI5', 'tN5'):
        v = np.asarray(globals()[k])
        print(k, v.shape, np.round(v.ravel()[:10], 2))
    z = np.load('/work/v10_parts.npz')
    print('v10_parts', {k: z[k].shape for k in z.files})
except Exception as ex:
    print('var err', repr(ex)[:150])
try:
    np.savez('/work/v10b_tensors.npz', **{n.replace('.', '__'): np.asarray(load(n)) for n in edited()})
    print('backup ok', edited())
except Exception as ex:
    print('bk err', repr(ex)[:150])
with open('/work/notes.md', 'a') as f:
    f.write("\n- RESTART@345: run298 = info dump (ids, truth lines, vars) + v10b backup /work/v10b_tensors.npz. "
            "PLAN: fwd factual both tuning (4x100) + dose_edit both sel100 on v10b. If dNLL>+.003: run299 rebuild weaker via build7; run300 reserve (fallback drop_edit).\n")
print('done')
```