The factual residual spreads are large for glucose (finger 14.2 mg/dL, lab 10.7) and smaller for MAP (arterial 5.1 mmHg, cuff 8.1). So injection noise of about 0.5 mg/dL on glucose costs only about 0.0006 nats per affected finger reading.

MAP is the tight constraint. Arterial noise of about 0.5 mmHg costs about 0.005 per arterial reading. Capturing the true residual effects pays some of this back: roughly 0.0016 per affected finger reading and 0.0027 per affected arterial reading.

That makes the b_in amplifier (route B) affordable for insulin, and for NE with a large token amplitude or a reduced gain. It is also simpler than a head-side decoder:
- z_post / z_in put a gated copy of u plus |v|·(rate − π̂(ctx)) on the SNR-optimal token directions;
- a rank-2 `core.b_in` term writes q·E into glucose- and MAP-specific state directions.

The largest piece is z_post. It needs two things that only a regression can give:
- a π̂(ctx) from gelu features, gated by drug presence;
- presence detectors and rate decoders on u.

So this run collects drug-bundle data from 1,500 training stays held out from the GBM. For each drug bundle it records ctx, e, u, the rates, presence flags and the GBM π̂, plus synthetic edited u's (x0–x3) for the u-side decoders.

I expect about 25k drug bundles in a few minutes. A large share of NaN π̂ would mean a join error between bundle times and the hourly table.

```python
import numpy as np, torch, time
samp2 = rng_.choice(np.setdiff1d(hold, samp), 1500, replace=False); S2 = set(samp2)
for c, cc in [('insulin', 8), ('norepinephrine', 9)]:
    m = HT[c].notna() & HT.stay_id.isin(S2) & (HT.t >= 0)
    for (s, t, p) in zip(HT.loc[m, 'stay_id'], HT.loc[m, 't'], POL[c].predict(HT.loc[m, FEATS])): PHI[(s, t, cc)] = p
def collect_drug(W, B_, sidv, nedit=True):
    times = [b[0] for b in B_]; rows = []; urows = []
    out = torch.tensor(X0row[:192], dtype=T)[None]; h = torch.tensor(X0row[192:960], dtype=T).reshape(1, 4, 192)
    ea, ua = split_fix(B_[0][1]); Ms = [set_enc(W, ea, 'encode')]; ts = [times[0]]
    for k in range(1, len(B_)):
        t = times[k]; dt = (t-times[k-1])*3600.0; atoms = B_[k][1]; ea, ua = split_fix(atoms)
        e = set_enc(W, ea, 'encode'); u = set_enc(W, ua, 'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt = torch.stack(Ms)[None]; ages = torch.tensor([(t-tj)*3600.0 for tj in ts], dtype=T)
        if ua:
            hf = h*torch.exp(-dt*torch.exp(W['core.flow.a']))
            ctx = out + bln(W, hf.mean(1), 'core.out_norm') + bmlp(W, torch.tensor([[math.log(dt)/10]], dtype=T), 'gap')
            raw = [a for a in atoms if a['act'] and a['kind']==0]; rr = {a['code']: rate_of(a) for a in raw}
            th = int(round(t-0.0833)); ph = {cc: PHI.get((sidv, th, cc), np.nan) for cc in rr}
            rows.append((sidv, t, 8 in rr, 9 in rr, rr.get(8, 0.0), rr.get(9, 0.0), ph.get(8, 0.0), ph.get(9, 0.0),
                         ctx[0].numpy().astype(np.float32), e.numpy().astype(np.float32), u.numpy().astype(np.float32)))
            if nedit:
                for dc in rr:
                    for f in FAC:
                        ue = set_enc(W, [fix_atom2(edit_atom(a, f) if a['code']==dc else a) for a in raw], 'encode_action')
                        urows.append((8 in rr, 9 in rr, rr.get(8, 0.0)*(f if dc==8 else 1), rr.get(9, 0.0)*(f if dc==9 else 1), ue.numpy().astype(np.float32)))
        out, h, _ = bstep(W, out, h, dt, e[None], u[None], Mt, ages)
    return rows, urows
t0c = time.time(); DR, UR = [], []; nst = 0
for s in samp2:
    r1, r2 = collect_drug(Wc, build_bundles(GB[s]), s, nedit=(nst < 400)); DR += r1; UR += r2; nst += 1
    if time.time()-t0c > 750: break
print('stays', nst, 'drug rows', len(DR), 'edited u rows', len(UR), 'time %.0fs' % (time.time()-t0c),
      'nan pihat', sum(1 for r in DR if (r[2] and np.isnan(r[6])) or (r[3] and np.isnan(r[7]))))
```