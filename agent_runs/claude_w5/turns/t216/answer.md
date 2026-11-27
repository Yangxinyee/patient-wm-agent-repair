The 4 slots carry identical statistics, so the slot axis looks degenerate (to check). If it is, one shared injection direction for all slots is natural.

More important: the LayerNorm scales vary a lot across drug bundles (CV 22% before the prelude FFN, 31% for P). The attention output (about 515) is comparable to the state (about 750), so in the normalized FFN input the drug signal's weight changes with the state's size.

Two things decide the design:
- how well "rate" and "policy mean" can be read linearly at each candidate site (prelude FFN input n3, versus the scale-free u token);
- whether those readouts respond linearly when the dose is actually doubled or halved, as in the evaluation.

Next I build x1/x2/x½ versions of every drug bundle by scaling the atoms' rate and dose and re-encoding them, then fit ridge decoders with a split by stay. The test is the slope of (decoded change) on (true change) for both x2 and x½. Slopes near 1 at a site make it usable; strongly sublinear slopes mean I must linearize the rate signal first.

```python
def isk0(a, code): return a.get('code') == code and str(a.get('kind')) == '0'
def scale_atoms(ats, code, f):
    out = []
    for a in ats:
        if isk0(a, code):
            a2 = dict(a); a2['rate'] = type(a['rate'])(float(a['rate'])*f); a2['dose'] = type(a['dose'])(float(a['dose'])*f); out.append(a2)
        else: out.append(a)
    return out
print('slots identical?', float((steps_all[sids_B[0]][5]['h0'][0]-steps_all[sids_B[0]][5]['h0'][1]).abs().max()),
      'core.slots row diff', float((W64['core.slots'][0]-W64['core.slots'][1]).abs().max()))
DB = []  # drug-bundle records
t0_ = time.time() if 'time' in globals() else None
import time
t0_ = time.time(); bad = 0
for s in sids_B:
    for k, d in enumerate(steps_all[s]):
        if (d['u'] - u_empty).norm() < 1e-6: continue
        ats = bundles_B[s][k][1]
        e1, u1 = enc2(ats)
        if (u1 - d['u']).norm() > 1e-6 * (1 + d['u'].norm()): bad += 1
        rec_ = dict(sid=s, k=k, rI=sum(float(a['rate']) for a in ats if isk0(a,'insulin')), rN=sum(float(a['rate']) for a in ats if isk0(a,'norepinephrine')),
                    hasI=any(isk0(a,'insulin') for a in ats), hasN=any(isk0(a,'norepinephrine') for a in ats),
                    st=torch.cat([d['out0'], d['h0'].mean(0)]).numpy(), u={}, n3={})
        for code, tagc in (('insulin','I'), ('norepinephrine','N')):
            for f in ((1.0, 2.0, 0.5) if (tagc=='I' and rec_['hasI']) or (tagc=='N' and rec_['hasN']) else (1.0,)):
                if f == 1.0 and 1.0 in rec_['u']: continue
                ats2 = scale_atoms(ats, code, f) if f != 1.0 else ats
                e2, u2 = enc2(ats2); d2 = dict(d); d2['e'] = e2; d2['u'] = u2
                R = {}; mystep(d2, R=R)
                key = 1.0 if f == 1.0 else (tagc, f)
                rec_['u'][key] = u2.numpy(); rec_['n3'][key] = R['n3'][0].numpy()
        DB.append(rec_)
print('drug bundles', len(DB), 'enc2 mismatches', bad, 'time', round(time.time()-t0_,1))
print('hasI', sum(r['hasI'] for r in DB), 'hasN', sum(r['hasN'] for r in DB), 'both', sum(r['hasI'] and r['hasN'] for r in DB))
```