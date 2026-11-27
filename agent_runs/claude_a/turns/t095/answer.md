The core has no built-in linear memory channel:
- The per-step carry factor a_bar is only 0.30–0.45 in every dimension.
- `c_out` is far from identity (relative distance 1.26), with singular values from 4.8 down to 0.
- At most about 20% of a state component survives an hour through the linear path.

Information persists only through the nonlinear blocks. Engineering a multi-hour dose memory inside the core by closed-form edits looks impractical.

The more workable route is to add the dose effect at the next reading and let the model's own feedback of drawn readings carry it forward, with the gain calibrated against the targets in offline rollouts.

That needs the dose to be readable from the state right after the dosing bundle. Next I'll measure how the head's query vector q at the next reading responds to scaling that bundle's rates. I'll compare insulin effects at glucose readings and norepinephrine effects at MAP readings, check how consistent the response direction is across instances, and compute a matched-filter readout's noise in rate units.
- A consistent direction with readout noise well below the rate's own spread (about 1 unit/h for insulin) would support a head-side readout edit.
- Inconsistent or noisy responses would mean I must strengthen the dose signal upstream, in the action encoder.

```python
def tf_views(df, s, P, mods, seed=0):
    at_s, bh_s = make_atoms(df, [s]); avl=[]; isact_s=None
    for md in mods:
        at_e = at_s.copy()
        for drug, key in [('insulin','ins'),('norepinephrine','ne')]:
            m_ = md.get(key, 1.0)
            if m_ != 1.0:
                msk = (at_e.event==drug); at_e.loc[msk,'rate'] *= m_; at_e.loc[msk,'dose'] *= m_
        av_, ia_ = atom_vectors(at_e, {}); avl.append(av_); isact_s = np.asarray(ia_).astype(bool)
    AVS = torch.stack(avl); nV = len(mods)
    rowsA = at_s['row'].values; ev = at_s['event'].values; hrs = at_s['hour'].values; rates = at_s['rate'].values
    mins = bh_s['minute'].values; dts = bh_s['dts'].values; nb = len(bh_s)
    h = torch.zeros(nV,4,192); o = torch.zeros(nV,192); Mb = torch.zeros(nV,0,192); Mm=[]; res=[]; last_rate = {'insulin':0.0,'norepinephrine':0.0}
    for j in range(nb):
        ia = np.where(rowsA==j)[0]; na = ia[~isact_s[ia]]; aa = ia[isact_s[ia]]
        dt = torch.full((nV,), 1.0 if j==0 else float(dts[j]))
        cx = ctx_fn(o, h, dt, P)
        evs = set(ev[na])
        for c in ('glucose_finger','map_arterial'):
            if j>0 and c in evs:
                pr_ = mlpP(cx, 'z_prior', P); view = cx + F.linear(pr_[:,:32], P['z_head.weight'], P['z_head.bias'])
                q = F.layer_norm(view + P['type_emb.weight'][1], (192,), P['query_norm.weight'], P['query_norm.bias'], 1e-5)
                res.append((s, float(bh_s['hour'].values[j]), c, last_rate['insulin' if c=='glucose_finger' else 'norepinephrine'], q.numpy().copy()))
        for drug in ('insulin','norepinephrine'):
            st_ = [i for i in aa if ev[i]==drug and at_s['kind'].values[i]==0]
            if len(st_): last_rate[drug] = float(rates[st_[0]])
            elif any(ev[i] in ('glucose_finger',) for i in na) and drug=='insulin': pass
        if len(na) and any(ev[i] in MEAS_CODES for i in na): last_rate = {k: (v if any(ev[i]==k for i in aa) else v) for k,v in last_rate.items()}
        e = ds_b(AVS[:, na], 'encode', P); u = ds_b(AVS[:, aa], 'encode_action', P)
        z = zpost_mean(cx, e, u, P)
        Mb = torch.cat([Mb, e[:,None]], 1)[:, -128:]; Mm = (Mm + [mins[j]])[-128:]
        h, o = core_step_b(h, e, u, z, dt, Mb, torch.tensor([(mins[j]-x)*60.0 for x in Mm], dtype=torch.float32), P)
        if len(aa)==0 and any(ev[i] in MEAS_CODES for i in na):
            pass
    return res
# lag-1 rate: rate of the infusion that started at hour-1+0.083 (use records)
def lag1_rate(df_s, drug, hr):
    r_ = df_s[(df_s.event==drug) & (np.abs(df_s.hour - (hr - 1 + 0.0833)) < 0.02)]['rate'].values
    return float(r_[0]) if len(r_) else 0.0
t0=time.time(); Dq = {'glucose_finger':[], 'map_arterial':[]}; Q0 = {'glucose_finger':[], 'map_arterial':[]}; RL = {'glucose_finger':[], 'map_arterial':[]}
for k_, s in enumerate(probe_stays[:100]):
    res = tf_views(tr, s, Po, [{}, {'ins':1.25, 'ne':1.25}])
    df_s = tr[tr.stay_id==s]
    for (s_, hr, c, _, q) in res:
        r1 = lag1_rate(df_s, 'insulin' if c=='glucose_finger' else 'norepinephrine', hr)
        Q0[c].append(q[0]); RL[c].append(r1)
        if r1 > 0: Dq[c].append((q[1]-q[0])/(0.25*r1))
print("time", round(time.time()-t0,1))
for c in Dq:
    D_ = np.array(Dq[c]); Qa = np.array(Q0[c]); m_ = D_.mean(0)
    cos_ = (D_ @ m_)/(np.linalg.norm(D_,axis=1)*np.linalg.norm(m_)+1e-12)
    Sig = np.cov(Qa.T) + 1e-4*np.eye(192); Si = np.linalg.solve(Sig, m_); snr = float(m_ @ Si)
    w_ = Si/snr; resp = D_ @ w_
    print(c, "n", len(D_), "|mean d|", round(float(np.linalg.norm(m_)),4), "cos median", round(float(np.median(cos_)),3), "readout noise sd (rate units)", round(1/math.sqrt(snr),3), "resp mean/sd", round(float(resp.mean()),3), round(float(resp.std()),3))
```