The atom-vector decomposition is exact, so a drawn reading can be embedded as base + `bins.value[b]` + `numeric[:,0]`·z(centre). The vocabulary confirms that the predicted mean is Σ p(b)·centre[b−1], with no renormalisation.

Next I'll write the offline `dose_edit` replica:
- the recorded prefix through hour 12;
- then 5 scenarios × 20 paths with common random numbers, the same as the tool;
- from hour 13 on, each reading's bin is drawn from the 4-draw-averaged head and its centre is fed back;
- doses are edited on drug atoms at hour ≥ 12;
- readouts are taken at the recorded glucose_finger / map_arterial readings in hours 13–24, pooled over reached stay-hours.

I'll validate it on the first 30 sel1 stays against forward 1's per-row factual means and shifts. Factual means within a couple of mg/dL and shifts of similar size (about −2 to −4 per row for insulin x2 on reached rows) would confirm the replica. A large factual mismatch would mean my sampling convention differs from the tool's.

```python
NUMV = {d_['code']: d_ for d_ in VOCAB['numeric'] if d_['attribute']=='value'}
MEAS_CODES = ['glucose_finger','glucose_lab','map_arterial','map_cuff']
ZC = {}
for c in MEAS_CODES:
    cen = centM[cid[c]].numpy().astype(np.float64)
    ZC[c] = torch.tensor(np.array([z32(c,'value',float(v)) for v in cen], dtype=np.float32))
    assert [fbin(c,'value',float(v)) for v in cen] == list(range(1,17)), c
SCEN = ['fact','insulin_x2','insulin_half','ne_x2','ne_half']
def prep_stay(df, s):
    at_s, bh_s = make_atoms(df, [s]); avs_s, isact_s = atom_vectors(at_s, {}); isact_s = np.asarray(isact_s).astype(bool)
    avl = []
    for sc in SCEN:
        if sc=='fact': avl.append(avs_s); continue
        drug = 'insulin' if sc.startswith('insulin') else 'norepinephrine'; m = 2.0 if sc.endswith('x2') else 0.5
        at_e = at_s.copy(); msk = (at_e.event==drug) & (at_e.hour >= 12-1e-6)
        at_e.loc[msk,'rate'] *= m; at_e.loc[msk,'dose'] *= m
        avl.append(atom_vectors(at_e, {})[0])
    return at_s, bh_s, isact_s, torch.stack(avl)
def rollout_stay(df, s, P, n_paths=20, seed=0, prep=None):
    at_s, bh_s, isact_s, AVS = prep if prep is not None else prep_stay(df, s)
    nS = len(SCEN); B = nS*n_paths
    rowsA = at_s['row'].values; ev = at_s['event'].values; hrs = at_s['hour'].values; vals = at_s['value'].values
    mins = bh_s['minute'].values; dts = bh_s['dts'].values; nb = len(bh_s)
    ismeas = np.isin(ev, MEAS_CODES) & ~isact_s & (at_s['kind'].values==0)
    numW0 = P['embed.numeric.weight'][:,0]; Wbv = P['embed.bins.value.weight']
    pre_rows = [r for r in range(nb) if mins[r] <= 720]
    h = torch.zeros(1,4,192); o = torch.zeros(1,192); Ml=[]; Mm=[]
    reg = {c: [0.0, 0.0, 0.0, 0] for c in MEAS_CODES}
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
    rec_ = []
    for r in [r for r in range(nb) if mins[r] > 720]:
        ia = np.where(rowsA==r)[0]; na = ia[~isact_s[ia]]; aa = ia[isact_s[ia]]
        dt = torch.full((B,), float(dts[r])); av_na = AVS[:, na].repeat_interleave(n_paths, 0).clone(); upd = {}
        if mins[r] >= 780:
            for jj, i in enumerate(na):
                if not ismeas[i]: continue
                c = ev[i]; has_, hr_, z_, b_ = regT[c]
                eps_ = torch.randn(n_paths, 4, 32, generator=g).repeat(nS,1,1)
                p_, m_ = head_value(o, h, dt, torch.full((B,),cid[c]), has_, (float(hrs[i])-hr_)*3600.0*has_, z_, b_, eps_, P)
                if c in ('glucose_finger','map_arterial'): rec_.append((float(hrs[i]), c, m_.view(nS, n_paths)))
                U = torch.rand(n_paths, generator=g).repeat(nS)
                pp = p_[:,1:]/p_[:,1:].sum(1,keepdim=True); bsel = (pp.cumsum(1) < U[:,None]).sum(1).clamp(max=15) + 1
                base_i = AVS[0, i] - Wbv[fbin(c,'value',float(vals[i]))] - numW0*float(z32(c,'value',float(vals[i])))
                av_na[:, jj] = base_i[None] + Wbv[bsel] + numW0[None]*ZC[c][bsel-1][:,None]
                upd[c] = (bsel, float(hrs[i]))
        e = ds_b(av_na, 'encode', P); u = ds_b(AVS[:, aa].repeat_interleave(n_paths, 0), 'encode_action', P)
        z = zpost_mean(ctx_fn(o, h, dt, P), e, u, P)
        Mb = torch.cat([Mb, e[:,None]], 1)[:, -128:]; Mm = (Mm + [mins[r]])[-128:]
        h, o = core_step_b(h, e, u, z, dt, Mb, torch.tensor([(mins[r]-x)*60.0 for x in Mm], dtype=torch.float32), P)
        for c, (bsel, hr) in upd.items(): regT[c] = [torch.ones(B), torch.full((B,),hr), ZC[c][bsel-1], bsel]
    first = {}
    for drug in ['insulin','norepinephrine']:
        st_ = hrs[(ev==drug) & (at_s['kind'].values==0) & (hrs>=12-1e-6)]; first[drug] = st_.min() if len(st_) else np.inf
    out_rows = []
    for (hr, c, mm) in rec_:
        if hr < 13 or hr > 24.01: continue
        eds = ['insulin_x2','insulin_half'] if c=='glucose_finger' else ['ne_x2','ne_half']
        for ed in eds:
            si = SCEN.index(ed); drug = 'insulin' if ed.startswith('insulin') else 'norepinephrine'
            d_ = (mm[si]-mm[0]); out_rows.append((s, int(round(hr)), ed, float(d_.mean()), float(d_.std()/math.sqrt(n_paths)), float(mm[0].mean()), hr > first[drug]))
    return pd.DataFrame(out_rows, columns=['stay_id','hour','edit','shift','shift_se','factual','reached'])
sel1 = json.load(open('/work/sel1.json'))['sel1']
t0 = time.time(); RO = pd.concat([rollout_stay(tu, s, Po, n_paths=20, seed=k_) for k_, s in enumerate(sel1[:30])]); print("time", round(time.time()-t0,1))
cmp_ = RO.merge(D1f.rename(columns={'shift':'shift_tool','factual':'fact_tool','reached':'reached_tool'}), on=['stay_id','hour','edit'], how='inner')
print("rows matched", len(cmp_), "of", len(RO), "| reached agree", float((cmp_.reached==cmp_.reached_tool).mean()))
for ed, g_ in cmp_.groupby('edit'):
    rr = g_[g_.reached_tool]
    print(ed, "fact rmse", round(float(np.sqrt(((g_.factual-g_.fact_tool)**2).mean())),2), "| reached n", len(rr), "mine", round(rr['shift'].mean(),3), "tool", round(rr['shift_tool'].mean(),3), "corr", round(np.corrcoef(rr['shift'], rr['shift_tool'])[0,1],2) if len(rr)>2 else None)
```