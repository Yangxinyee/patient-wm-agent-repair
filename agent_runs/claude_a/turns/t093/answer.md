The model passes a change in a drawn reading forward quite persistently. A +20 mg/dL shift in one finger glucose moves the later predicted means by 8.9, 8.4, 7.8, 7.6, … 6.6 at lags 1–11 (ratio about 0.44 falling to 0.33). A +10 mmHg MAP shift gives 4.6, 3.7, 3.25, then about 3.0 for many hours.

So a dose effect delivered only at lag 1 would be carried forward through the model's own drawn readings, with a long tail. In the true effect kernel, the response peaks at lag 2 and fades by lag 10.

The right correction depends on how the model already uses dose history when readings are held at their recorded values. The model currently ignores insulin magnitude at lag 1, but the lag profile and the norepinephrine terms are unknown. Next I'll write a batched teacher-forced predictor that accepts dose modifications. I'll randomly perturb every infusion's rate (log-normal, independent per hour) and regress the change in predicted means on the rate changes at lags 1–10, separately for insulin→glucose and norepinephrine→MAP.

This gives the model's direct dose kernel. I'll compare it with the direct kernel implied by the records (about −2.1, −1.5, −0.5, then positive at lags 4–8 for insulin). That tells me what the edit has to add at each lag. I expect near-zero insulin sensitivity at all lags; for norepinephrine, a small positive lag-1 term with negative later terms would explain the wrong-signed rollout.

```python
def tf_predict(df, s, P, mods, nz=8, seed=0):
    """teacher-forced predicted means at recorded readings, for several dose-modification variants.
       mods: list of dicts {'ins': array of multipliers per insulin START atom (by order) or None, 'ne': ...}"""
    at_s, bh_s = make_atoms(df, [s]); isact_s = None
    avl = []
    for md in mods:
        at_e = at_s.copy()
        for drug, key in [('insulin','ins'),('norepinephrine','ne')]:
            mlt = md.get(key)
            if mlt is None: continue
            starts = at_e.index[(at_e.event==drug)&(at_e.kind==0)].tolist()
            for jj, ix in enumerate(starts):
                hr0 = at_e.at[ix,'hour']; m_ = mlt[jj]
                msk = (at_e.event==drug) & (np.abs(at_e.hour - hr0) < 1e-6) & (at_e.kind==0) | (at_e.event==drug) & (np.abs(at_e.hour - (hr0+1.0)) < 1e-3) & (at_e.kind==2)
                at_e.loc[msk,'rate'] = at_s.loc[msk,'rate']*m_; at_e.loc[msk,'dose'] = at_s.loc[msk,'dose']*m_
        av_, ia_ = atom_vectors(at_e, {}); avl.append(av_); isact_s = np.asarray(ia_).astype(bool)
    AVS = torch.stack(avl); nV = len(mods)
    rowsA = at_s['row'].values; ev = at_s['event'].values; hrs = at_s['hour'].values; vals = at_s['value'].values
    mins = bh_s['minute'].values; dts = bh_s['dts'].values; nb = len(bh_s)
    ismeas = np.isin(ev, MEAS_CODES) & ~isact_s & (at_s['kind'].values==0)
    g = torch.Generator().manual_seed(seed)
    h = torch.zeros(nV,4,192); o = torch.zeros(nV,192); Mb = torch.zeros(nV,0,192); Mm=[]
    reg = {c: [0.0, 0.0, 0.0, 0] for c in MEAS_CODES}; out_ = []
    for j in range(nb):
        ia = np.where(rowsA==j)[0]; na = ia[~isact_s[ia]]; aa = ia[isact_s[ia]]
        dt = torch.full((nV,), 1.0 if j==0 else float(dts[j]))
        if j>0:
            for i in na:
                if ismeas[i] and ev[i] in ('glucose_finger','map_arterial'):
                    c = ev[i]; eps_ = torch.randn(1, nz, 32, generator=g).repeat(nV,1,1)
                    has_ = torch.full((nV,), reg[c][0]); age_ = torch.full((nV,), (float(hrs[i])-reg[c][1])*3600.0*reg[c][0])
                    p_, m_ = head_value(o, h, dt, torch.full((nV,),cid[c]), has_, age_, torch.full((nV,),reg[c][2]), torch.full((nV,),reg[c][3],dtype=torch.long), eps_, P)
                    out_.append((s, float(hrs[i]), c, float(vals[i]), m_.numpy()))
        e = ds_b(AVS[:, na], 'encode', P); u = ds_b(AVS[:, aa], 'encode_action', P)
        z = zpost_mean(ctx_fn(o, h, dt, P), e, u, P)
        Mb = torch.cat([Mb, e[:,None]], 1)[:, -128:]; Mm = (Mm + [mins[j]])[-128:]
        h, o = core_step_b(h, e, u, z, dt, Mb, torch.tensor([(mins[j]-x)*60.0 for x in Mm], dtype=torch.float32), P)
        for i in na:
            if ismeas[i]: reg[ev[i]] = [1.0, float(hrs[i]), float(z32(ev[i],'value',float(vals[i]))), int(fbin(ev[i],'value',float(vals[i])))]
    return out_, at_s
rng_ = np.random.default_rng(5)
tr_stays = list(pd.unique(tr['stay_id']))
probe_stays = list(rng_.choice(tr_stays, 150, replace=False))
rowsP = []; t0 = time.time()
for k_, s in enumerate(probe_stays):
    at0 = tr[tr.stay_id==s]; ni = int(((at0.event=='insulin')).sum()); nn = int(((at0.event=='norepinephrine')).sum())
    mi = np.exp(rng_.normal(0, 0.4, ni)); mn = np.exp(rng_.normal(0, 0.4, nn))
    out_, at_s = tf_predict(tr, s, Po, [{}, {'ins': mi, 'ne': mn}], nz=8, seed=k_)
    ins_t = at0[at0.event=='insulin'][['hour','rate']].values; ne_t = at0[at0.event=='norepinephrine'][['hour','rate']].values
    for (s_, hr, c, v, mm) in out_:
        row = [s_, hr, c, mm[1]-mm[0], mm[0]]
        for lag in range(1, 11):
            di = [ (mi[jj]-1)*ins_t[jj,1] for jj in range(ni) if abs(ins_t[jj,0] - (hr - lag + 0.0833)) < 0.02 ]
            dn = [ (mn[jj]-1)*ne_t[jj,1] for jj in range(nn) if abs(ne_t[jj,0] - (hr - lag + 0.0833)) < 0.02 ]
            row += [sum(di), sum(dn)]
        rowsP.append(row)
cols = ['stay','hour','code','dpred','pred'] + sum([[f'di{l}', f'dn{l}'] for l in range(1,11)], [])
PR = pd.DataFrame(rowsP, columns=cols); print("time", round(time.time()-t0,1), "rows", len(PR))
for c, nm_ in [('glucose_finger','di'), ('map_arterial','dn'), ('glucose_finger','dn'), ('map_arterial','di')]:
    g_ = PR[PR.code==c]; X_ = np.column_stack([g_[[f'{nm_}{l}' for l in range(1,11)]].values, g_[[f'{"dn" if nm_=="di" else "di"}{l}' for l in range(1,11)]].values])
    b_, *_ = np.linalg.lstsq(X_, g_['dpred'].values, rcond=None)
    print(c, "<-", nm_, np.round(b_[:10], 3))
```