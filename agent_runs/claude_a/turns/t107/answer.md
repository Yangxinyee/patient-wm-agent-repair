The builders reproduce the sel2 inputs exactly: same bundle order, and e/u errors of 0.0. They are fast, about 0.06 s per 100 stays.

The bundle table's `dts` column uses whole minutes (3300 s), while the states tool used hour-based gaps (3300.12 s). I will compute gaps from hours so they match.

**This run.**
1. Write a general dataset builder: bundles, e and u, time gaps, and the per-reading register features (previous bundle, gap, code, last value with its age, z-score and bin, observed bin).
2. Validate it against the sel2 arrays that already reproduce the tool.
3. Generalise the teacher-forced pass to any dataset, and check that the original NLL comes out the same on sel2.

**How I will read it.** Identical register features and the same NLL (1.40449) confirm the builder, and I can then scale to thousands of training stays. A mismatch would most likely point to the z-score or bin convention for the last value.

```python
MEAS = ['glucose_finger', 'glucose_lab', 'map_arterial', 'map_cuff']
def build_ds(df, stays, seed=0):
    at_, bd_ = make_atoms(df, list(stays))
    e_, u_ = encode_bundles(at_, list(range(len(bd_))))
    hrs = bd_['hour'].values; prev = bd_['prev'].values
    dts = np.where(prev >= 0, (hrs - hrs[np.maximum(prev, 0)]) * 3600.0, 1.0)
    srows = [np.where(bd_['stay_id'].values == s_)[0] for s_ in pd.unique(bd_['stay_id'].values)]
    key_ = pd.Series(np.arange(len(bd_)), index=pd.MultiIndex.from_arrays([bd_['stay_id'].values, bd_['minute'].values]))
    rd = at_[(at_['kind'] == 0) & at_['event'].isin(MEAS) & at_['value'].notna()].copy()
    rd = rd.sort_values(['stay_id', 'hour', 'event']).reset_index(drop=True)
    rd['row_self'] = key_.reindex(pd.MultiIndex.from_arrays([rd['stay_id'].values, rd['minute'].values])).values
    rd['row_prev'] = prev[rd['row_self'].values]
    g_ = rd.groupby(['stay_id', 'event'])
    rd['lastv'] = g_['value'].shift(1); rd['lasth'] = g_['hour'].shift(1)
    has = rd['lastv'].notna().values
    zlast = np.array([zsc_v(ev_, v_) if h_ else 0.0 for ev_, v_, h_ in zip(rd['event'], rd['lastv'].fillna(0), has)], dtype=np.float32)
    lvb_ = np.array([fbin_v(ev_, v_) if h_ else 0 for ev_, v_, h_ in zip(rd['event'], rd['lastv'].fillna(0), has)], dtype=np.int64)
    ob_ = np.array([fbin_v(ev_, v_) for ev_, v_ in zip(rd['event'], rd['value'])], dtype=np.int64)
    rs = rd['row_self'].values.astype(int)
    R_ = dict(prevrow=rd['row_prev'].values.astype(int), dtv=torch.tensor(dts[rs], dtype=torch.float32),
              cidx=torch.tensor([cid[x] for x in rd['event']]), hasl=torch.tensor(has.astype(np.float32)),
              ages=torch.tensor(np.where(has, (rd['hour'] - rd['lasth'].fillna(0)).values * 3600.0, 0.0), dtype=torch.float32),
              zl=torch.tensor(zlast), lvbt=torch.tensor(lvb_), obs=torch.tensor(ob_))
    rng_l = np.random.default_rng(seed)
    R_['eps'] = torch.tensor(rng_l.standard_normal((len(rd), 4, 32)), dtype=torch.float32)
    return dict(at=at_, bd=bd_, e=e_, u=u_, dts=torch.tensor(dts, dtype=torch.float32), tsec=torch.tensor(hrs * 3600.0, dtype=torch.float32), srows=srows, rd=rd, R=R_)
def tf_states_ds(P, ds):
    N = len(ds['bd']); Hn = torch.zeros(N, 4, 192); On = torch.zeros(N, 192)
    srows = ds['srows']; L = max(len(r) for r in srows)
    with torch.no_grad():
        for k in range(L):
            idx = [r for r in srows if len(r) > k]
            rows_k = np.array([r[k] for r in idx])
            if k == 0:
                hprev = torch.zeros(len(rows_k), 4, 192); oprev = torch.zeros(len(rows_k), 192)
            else:
                pr = np.array([r[k-1] for r in idx]); hprev = Hn[pr]; oprev = On[pr]
            dt = ds['dts'][rows_k]
            ctx = ctx_fn(oprev, hprev, dt, P); z = zpost_mean(ctx, ds['e'][rows_k], ds['u'][rows_k], P)
            mem = torch.tensor(np.stack([r[:k+1] for r in idx]))
            t = ds['tsec'][mem]; ages_ = (t[:, -1:] - t).clamp(min=0)
            y, o = step_b(hprev, ds['e'][rows_k], ds['u'][rows_k], z, dt, ds['e'][mem], ages_, torch.ones(len(rows_k), k+1, dtype=torch.bool), P)
            Hn[rows_k] = y; On[rows_k] = o
    return Hn, On
def tf_eval(P, ds, ret_states=False):
    Hn, On = tf_states_ds(P, ds); R_ = ds['R']
    ph, mh = head_value(On[R_['prevrow']], Hn[R_['prevrow']], R_['dtv'], R_['cidx'], R_['hasl'], R_['ages'], R_['zl'], R_['lvbt'], R_['eps'], P)
    nll = -torch.log(ph[torch.arange(len(R_['obs'])), R_['obs']])
    return (nll, ph, mh, Hn, On) if ret_states else (nll, ph, mh)
t0_ = time.time()
ds2 = build_ds(tr, stays_SB)
print('build time', round(time.time() - t0_, 2), ' n readings', len(ds2['rd']), 'vs', len(obs_b))
R2_ = ds2['R']
if len(ds2['rd']) == len(obs_b):
    print('prevrow eq', (R2_['prevrow'] == prevrow).mean(), ' dtv maxdiff', (R2_['dtv'] - dtv).abs().max().item(), ' cidx eq', (R2_['cidx'] == cidx).float().mean().item(),
          ' hasl', (R2_['hasl'] == hasl).float().mean().item(), ' ages maxdiff', (R2_['ages'] - ages).abs().max().item(), ' zl maxdiff', (R2_['zl'] - zl).abs().max().item(),
          ' lvb eq', (R2_['lvbt'].numpy() == lvb).mean(), ' obs eq', (R2_['obs'].numpy() == obs_b).mean())
    R2_['eps'] = eps4
nll_ds2, _, _ = tf_eval(PO, ds2)
print('NLL via ds pipeline', nll_ds2.mean().item(), ' (target 1.404487)', round(time.time() - t0_, 2), 's')
```