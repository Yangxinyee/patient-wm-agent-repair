`build_bundles(ds)` turns one stay's rows into time-sorted bundles of atoms, the same structure BUN holds. So I can replay training stays exactly.

This run collects the decoder's training data on 300 training stays held out from the policy fit. For each I run the exact teacher-forced replica of the original model and record:
- the view base before every bundle;
- the residual target r = rate − π̂(GBM) when the previous bundle was an insulin or NE infusion, and zero otherwise.

At every drug bundle I also branch with the drug's rate edited (x0, x1/2, x1.5, x2, x3) and record the next bundle's base with the edited residual. That way the decoder learns the range the dose interventions will produce.

I also check that my u-building for branches matches the replica's own u. If it doesn't, the edited branches would be inconsistent with the rollout and must be fixed first.

```python
import numpy as np, torch, time, pandas as pd
DF = pd.read_csv('/data/train.csv'); GB = dict(tuple(DF.groupby('stay_id')))
hold = sid[22000:]; rng_ = np.random.default_rng(1); samp = rng_.choice(hold, 300, replace=False)
PHI = {}
for c, cc in [('insulin', 8), ('norepinephrine', 9)]:
    m = HT[c].notna() & HT.stay_id.isin(set(samp)) & (HT.t >= 0)
    for (s, t, p) in zip(HT.loc[m, 'stay_id'], HT.loc[m, 't'], POL[c].predict(HT.loc[m, FEATS])): PHI[(s, t, cc)] = p
def base_of(W, out, h, dt):
    hf = h*torch.exp(-dt*torch.exp(W['core.flow.a']))
    return out + bln(W, hf.mean(1), 'core.out_norm') + bmlp(W, torch.tensor([[math.log(dt)/10]], dtype=T), 'gap')
def rate_of(a):
    nmd = DRUGN[a['code']]; return a['num'][4]*NUMV[(nmd, 'rate')]['iqr'] + NUMV[(nmd, 'rate')]['median']
FAC = [0.0, 0.5, 1.5, 2.0, 3.0]
def collect_stay(W, B_, sidv):
    times = [b[0] for b in B_]; rows = []
    out = torch.tensor(X0row[:192], dtype=T)[None]; h = torch.tensor(X0row[192:960], dtype=T).reshape(1, 4, 192)
    ea, ua = split_fix(B_[0][1]); Ms = [set_enc(W, ea, 'encode')]; ts = [times[0]]; prev = None
    for k in range(1, len(B_)):
        t = times[k]; dt = (t-times[k-1])*3600.0; atoms = B_[k][1]
        b0 = base_of(W, out, h, dt)[0].numpy().astype(np.float32)
        if prev is None: rows.append((sidv, t, 0, 0.0, 0.0, 1.0, 0, b0))
        else:
            for (ri_, rn_, f_, dc_, bb) in prev:
                bb2 = base_of(W, bb[0], bb[1], dt)[0].numpy().astype(np.float32)
                rows.append((sidv, t, 1, ri_, rn_, f_, dc_, bb2))
        ea, ua = split_fix(atoms)
        e = set_enc(W, ea, 'encode'); u = set_enc(W, ua, 'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt = torch.stack(Ms)[None]; ages = torch.tensor([(t-tj)*3600.0 for tj in ts], dtype=T)
        prev = None
        if ua:
            raw = [a for a in atoms if a['act'] and a['kind']==0]
            th = int(round(t-0.0833)); ph = {a['code']: PHI.get((sidv, th, a['code']), np.nan) for a in raw}
            rr = {a['code']: rate_of(a) for a in raw}
            var = [(None, 1.0)] + [(dc, f) for dc in rr for f in FAC]
            us = [u] + [set_enc(W, [fix_atom2(edit_atom(a, f) if a['code']==dc else a) for a in raw], 'encode_action') for dc, f in var[1:]]
            U_ = torch.stack(us); Bt = len(us)
            o2, h2, _ = bstep(W, out.repeat(Bt, 1), h.repeat(Bt, 1, 1), dt, e[None].repeat(Bt, 1), U_, Mt.repeat(Bt, 1, 1), ages)
            prev = []
            for j, (dc, f) in enumerate(var):
                ri_ = (rr[8]*(f if dc==8 else 1.0) - ph[8]) if 8 in rr else 0.0
                rn_ = (rr[9]*(f if dc==9 else 1.0) - ph[9]) if 9 in rr else 0.0
                prev.append((ri_, rn_, f, dc or 0, (o2[j:j+1], h2[j:j+1])))
            out, h = o2[0:1], h2[0:1]
        else:
            out, h, _ = bstep(W, out, h, dt, e[None], u[None], Mt, ages)
    return rows
B_t = build_bundles(GB[samp[0]]); raw0 = next(b for b in B_t if any(a['act'] and a['kind']==0 for a in b[1]))
ra = [a for a in raw0[1] if a['act'] and a['kind']==0]
print('u consistency', float((set_enc(Wc, [fix_atom2(a) for a in ra], 'encode_action') - set_enc(Wc, split_fix(raw0[1])[1], 'encode_action')).abs().max()))
t0c = time.time(); DEC = []
for i_, s in enumerate(samp):
    DEC += collect_stay(Wc, build_bundles(GB[s]), s)
    if time.time()-t0c > 700: print('stopped at', i_); break
print('rows', len(DEC), 'time %.0fs' % (time.time()-t0c), 'nan pihat rows', sum(1 for r in DEC if np.isnan(r[3]) or np.isnan(r[4])))
```