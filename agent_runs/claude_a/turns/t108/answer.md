The build failed because `zsc_v` (and probably `fbin_v`) take arrays of values, not single numbers. This is a calling-convention bug, not a problem with the approach.

I will compute the z-scores and bins for each code in one vectorised call, then rerun the same validation against the sel2 arrays. The confirmation is the same as before: identical register features and an NLL of 1.40449.

```python
def zf_vec(events, vals, mask):
    z_ = np.zeros(len(vals), dtype=np.float32); b_ = np.zeros(len(vals), dtype=np.int64)
    ev_arr = np.asarray(events); v_arr = np.asarray(vals, dtype=float)
    for ev_ in np.unique(ev_arr):
        m_ = (ev_arr == ev_) & mask
        if m_.sum() == 0: continue
        z_[m_] = np.asarray(zsc_v(ev_, v_arr[m_]), dtype=np.float32)
        b_[m_] = np.asarray(fbin_v(ev_, v_arr[m_]), dtype=np.int64)
    return z_, b_
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
    zlast, lvb_ = zf_vec(rd['event'].values, rd['lastv'].fillna(0).values, has)
    _, ob_ = zf_vec(rd['event'].values, rd['value'].values, np.ones(len(rd), bool))
    rs = rd['row_self'].values.astype(int)
    R_ = dict(prevrow=rd['row_prev'].values.astype(int), dtv=torch.tensor(dts[rs], dtype=torch.float32),
              cidx=torch.tensor([cid[x] for x in rd['event']]), hasl=torch.tensor(has.astype(np.float32)),
              ages=torch.tensor(np.where(has, (rd['hour'] - rd['lasth'].fillna(0)).values * 3600.0, 0.0), dtype=torch.float32),
              zl=torch.tensor(zlast), lvbt=torch.tensor(lvb_), obs=torch.tensor(ob_))
    rng_l = np.random.default_rng(seed)
    R_['eps'] = torch.tensor(rng_l.standard_normal((len(rd), 4, 32)), dtype=torch.float32)
    return dict(at=at_, bd=bd_, e=e_, u=u_, dts=torch.tensor(dts, dtype=torch.float32), tsec=torch.tensor(hrs * 3600.0, dtype=torch.float32), srows=srows, rd=rd, R=R_)
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