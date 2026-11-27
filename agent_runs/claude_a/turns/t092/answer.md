The offline rollout reproduces the tool's dose-edit readouts on the 30 stays:
- insulin x2: −1.99 (tool −2.03)
- insulin x½: +1.47 (tool +1.42)
- norepinephrine x2: −0.63 (tool −0.78)
- norepinephrine x½: −0.44 (tool −0.51)

Per-row correlations are 0.90–0.99, and the reached flags agree exactly. 100 stays take about 26 s, so I can test edits offline and save forward passes for confirmation.

Before designing the edit, I need to know how the model itself carries a change in a reading forward. In a rollout, any dose effect I add at lag 1 is passed on to later hours through the model's own drawn readings. To find out, I'll generalise the rollout so that a scenario can shift one drawn reading: +20 mg/dL at the first finger glucose from hour 13, or +10 mmHg at the first arterial MAP. I'll then measure the response of the later predicted means by lag.
- Near-1 persistence (ρ ≈ 0.8–1) would mean a lag-1 dose effect plus the model's own propagation could already give most of the cumulative effect.
- Fast-decaying ρ would mean I must build a multi-hour dose memory into the state.

```python
with open('/work/notes.md','a') as fh:
    fh.write("- Run87: OFFLINE ROLLOUT REPLICA rollout_stay(df,s,P,n_paths,seed) validated vs fwd1 on 30 sel1 stays: ins_x2 -1.99 (tool -2.03), half 1.47 (1.42), ne_x2 -0.63 (-0.78), ne_half -0.44 (-0.51); row corr .9-.99. ~0.26 s/stay.\n")
def rollout_gen(df, s, P, specs, n_paths=20, seed=0, prep_cache=None):
    """specs: list of dicts {'ins':m,'ne':m,'shift':(code, delta) or None}; spec 0 is reference. shift applies at first reading of code at hour>=13."""
    at_s, bh_s = make_atoms(df, [s]); avs_s, isact_s = atom_vectors(at_s, {}); isact_s = np.asarray(isact_s).astype(bool)
    avl = []
    for sp_ in specs:
        at_e = at_s.copy()
        for drug, key in [('insulin','ins'),('norepinephrine','ne')]:
            m = sp_.get(key, 1.0)
            if m != 1.0:
                msk = (at_e.event==drug) & (at_e.hour >= 12-1e-6); at_e.loc[msk,'rate'] *= m; at_e.loc[msk,'dose'] *= m
        avl.append(avs_s if all(sp_.get(k,1.0)==1.0 for k in ('ins','ne')) else atom_vectors(at_e, {})[0])
    AVS = torch.stack(avl); nS = len(specs); B = nS*n_paths
    rowsA = at_s['row'].values; ev = at_s['event'].values; hrs = at_s['hour'].values; vals = at_s['value'].values
    mins = bh_s['minute'].values; dts = bh_s['dts'].values; nb = len(bh_s)
    ismeas = np.isin(ev, MEAS_CODES) & ~isact_s & (at_s['kind'].values==0)
    numW0 = P['embed.numeric.weight'][:,0]; Wbv = P['embed.bins.value.weight']
    pre_rows = [r for r in range(nb) if mins[r] <= 720]
    h = torch.zeros(1,4,192); o = torch.zeros(1,192); Ml=[]; Mm=[]; reg = {c: [0.0, 0.0, 0.0, 0] for c in MEAS_CODES}
    for j, r in enumerate(pre_rows):
        ia = np.where(rowsA==r)[0]; na = ia[~isact_s[ia]]; aa = ia[isact_s[ia]]
        e = ds_b(AVS[0, na][None], 'encode', P); u = ds_b(AVS[0, aa][None], 'encode_action', P)
        dt = torch.tensor([1.0 if j==0 else float(dts[r])]); z = zpost_mean(ctx_fn(o, h, dt, P), e, u, P)
        Ml.append(e[0]); Mm.append(mins[r]); Ml=Ml[-128:]; Mm=Mm[-128:]
        h, o = core_step_b(h, e, u, z, dt, torch.stack(Ml)[None], torch.tensor([(mins[r]-x)*60.0 for x in Mm]), P)
        for i in na:
            if ismeas[i]: reg[ev[i]] = [1.0, float(hrs[i]), float(z32(ev[i],'value',float(vals[i]))), int(fbin(ev[i],'value',float(vals[i])))]
    g = torch.Generator().manual_seed(seed)
    h = h.expand(B,4,192).clone(); o = o.expand(B,192).clone(); Mb = torch.stack(Ml)[None].expand(B,-1,-1).clone()
    regT = {c: [torch.full((B,),reg[c][0]), torch.full((B,),reg[c][1]), torch.full((B,),reg[c][2]), torch.full((B,),reg[c][3],dtype=torch.long)] for c in MEAS_CODES}
    done_shift = set(); rec_ = []
    for r in [r for r in range(nb) if mins[r] > 720]:
        ia = np.where(rowsA==r)[0]; na = ia[~isact_s[ia]]; aa = ia[isact_s[ia]]
        dt = torch.full((B,), float(dts[r])); av_na = AVS[:, na].repeat_interleave(n_paths, 0).clone(); upd = {}
        if mins[r] >= 780:
            for jj, i in enumerate(na):
                if not ismeas[i]: continue
                c = ev[i]; has_, hr_, z_, b_ = regT[c]
                eps_ = torch.randn(n_paths, 4, 32, generator=g).repeat(nS,1,1)
                p_, m_ = head_value(o, h, dt, torch.full((B,),cid[c]), has_, (float(hrs[i])-hr_)*3600.0*has_, z_, b_, eps_, P)
                rec_.append((float(hrs[i]), c, m_.view(nS, n_paths)))
                U = torch.rand(n_paths, generator=g).repeat(nS)
                pp = p_[:,1:]/p_[:,1:].sum(1,keepdim=True); bsel = (pp.cumsum(1) < U[:,None]).sum(1).clamp(max=15) + 1
                vsel = centM[cid[c]][bsel-1].clone()
                for si, sp_ in enumerate(specs):
                    sh = sp_.get('shift')
                    if sh is not None and sh[0]==c and (si, c) not in done_shift:
                        vsel[si*n_paths:(si+1)*n_paths] += sh[1]; done_shift.add((si, c))
                bnew = torch.tensor([fbin(c,'value',float(v)) for v in vsel]); znew = torch.tensor([float(z32(c,'value',float(v))) for v in vsel])
                base_i = AVS[0, i] - Wbv[fbin(c,'value',float(vals[i]))] - numW0*float(z32(c,'value',float(vals[i])))
                av_na[:, jj] = base_i[None] + Wbv[bnew] + numW0[None]*znew[:,None]
                upd[c] = (bnew, znew, float(hrs[i]))
        e = ds_b(av_na, 'encode', P); u = ds_b(AVS[:, aa].repeat_interleave(n_paths, 0), 'encode_action', P)
        z = zpost_mean(ctx_fn(o, h, dt, P), e, u, P)
        Mb = torch.cat([Mb, e[:,None]], 1)[:, -128:]; Mm = (Mm + [mins[r]])[-128:]
        h, o = core_step_b(h, e, u, z, dt, Mb, torch.tensor([(mins[r]-x)*60.0 for x in Mm], dtype=torch.float32), P)
        for c, (bnew, znew, hr) in upd.items(): regT[c] = [torch.ones(B), torch.full((B,),hr), znew, bnew]
    first = {}
    for drug in ['insulin','norepinephrine']:
        st_ = hrs[(ev==drug) & (at_s['kind'].values==0) & (hrs>=12-1e-6)]; first[drug] = st_.min() if len(st_) else np.inf
    return rec_, first
specs_imp = [{}, {'shift':('glucose_finger', 20.0)}, {'shift':('map_arterial', 10.0)}]
imp_rows = []
t0=time.time()
for k_, s in enumerate(sel1):
    rec_, first = rollout_gen(tu, s, Po, specs_imp, n_paths=10, seed=k_)
    t_g = [hr for (hr,c,mm) in rec_ if c=='glucose_finger']; t_m = [hr for (hr,c,mm) in rec_ if c=='map_arterial']
    for (hr, c, mm) in rec_:
        if c=='glucose_finger' and t_g: imp_rows.append((s, c, hr - t_g[0], float((mm[1]-mm[0]).mean())))
        if c=='map_arterial' and t_m: imp_rows.append((s, c, hr - t_m[0], float((mm[2]-mm[0]).mean())))
IMP = pd.DataFrame(imp_rows, columns=['stay','code','lag','resp']); print("time", round(time.time()-t0,1))
print(IMP[IMP.lag>0].groupby(['code', IMP.lag.round()])['resp'].agg(['mean','count']).unstack(0).round(2).head(12))
```