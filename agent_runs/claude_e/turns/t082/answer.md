The states for 4629 bundles across selection A have arrived. Next I will build a general reading-level pipeline and save it to /work/lib.py: rows for every reading, plus the local head's predicted mean and bin NLL from 16 z-draws. I will then apply the identical one-step regression to the model's predicted glucose means for these stays.

If the model's dose-deviation coefficients come out near zero (or slightly positive, as the 38-stay estimate suggested), the missing effect γ = β_true − β_model is about the full true coefficient.

```python
lib3 = r'''
KEYS = {'glucose_finger':'gf','glucose_lab':'gl','map_arterial':'ma','map_cuff':'mc'}
CODE = {'glucose_finger':6,'glucose_lab':10,'map_arterial':7,'map_cuff':5}
def build_reading_rows(npz, G):
    zz = np.load(npz, allow_pickle=True)
    Xs = torch.as_tensor(np.array(zz['x']), dtype=torch.float64)
    st = pd.DataFrame({'stay': zz['stay_id'], 'hour': zz['hour']}); st['row'] = np.arange(len(st))
    rows = []
    for sid, g in st.groupby('stay', sort=False):
        hrs_ = g.hour.values; rws = g.row.values
        for ev, k in KEYS.items():
            ser = G[k].loc[sid]; ser = ser[ser.notna()]
            m_, iq_, edges = med[ev]
            last_t = None; last_v = None
            for t, v in ser.items():
                i = np.searchsorted(hrs_, t-1e-6) - 1
                if i < 0: 
                    last_t, last_v = t, v; continue
                if last_t is None: hs, age, zl_, lvb_ = 0.0, 0.0, 0.0, 0
                else: hs, age, zl_, lvb_ = 1.0, (t-last_t)*3600, float(np.clip((last_v-m_)/iq_,-10,10)), 1+int(np.sum(np.array(edges) < last_v))
                rows.append((sid, t, ev, CODE[ev], rws[i], (t-hrs_[i])*3600, hs, age, zl_, lvb_, v, 1+int(np.sum(np.array(edges) < v))))
                last_t, last_v = t, v
    R_ = pd.DataFrame(rows, columns=['stay','hour','event','code','prev','dt','has','age','zl','lvb','obs','obin'])
    return Xs, R_
def head_eval(Xs, R_, nz=16, seed=0, W=None, chunk=4000):
    W = W or {}
    pm_all, nll_all, P_all = [], [], []
    for a in range(0, len(R_), chunk):
        r = R_.iloc[a:a+chunk]
        Q = q_from_rows(Xs, torch.as_tensor(r.prev.values), torch.as_tensor(r.dt.values, dtype=torch.float64), nz=nz, seed=seed+a)
        f = reg_feats(torch.as_tensor(r.has.values), torch.as_tensor(r.age.values), torch.as_tensor(r.zl.values))
        code = torch.as_tensor(r.code.values); lvb = torch.as_tensor(r.lvb.values)
        lg, wc, hid = value_logits(Q, code, lvb, f, **W)
        p = torch.softmax(lg, -1).mean(0)
        cent = torch.stack([CENT[int(c)] for c in r.code.values])
        pm_all.append((p[:,1:]*cent).sum(1).numpy())
        nll_all.append(-torch.log(p[torch.arange(len(r)), torch.as_tensor(r.obin.values)]).numpy())
        P_all.append(p.numpy())
    return np.concatenate(pm_all), np.concatenate(nll_all), np.concatenate(P_all)
'''
open('/work/lib.py','a').write(lib3); exec(lib3)
XA, RA = build_reading_rows('/work/fwd_4/states_original.npz', Gtr)
pmA, nllA, PA = head_eval(XA, RA, nz=16)
RA['pm'] = pmA; RA['nll'] = nllA
print(RA.shape, RA.groupby('event').nll.mean().round(4).to_dict(), ' overall', RA.nll.mean().round(4))
# model one-step dev coefficients on the same specification
DA = build_ins_design(Gtr, selA)
DA['stay'] = np.array(selA)[DA.s.values]
DA = DA.merge(RA[RA.event=='glucose_finger'][['stay','hour','pm']], on=['stay','hour'], how='inner')
bt, set_, _ = ins_regress(DA, 'y'); bm, sem, rsm = ins_regress(DA, 'pm')
print('rows', len(DA), '\n true(obs) these stays', bt.round(2), set_.round(2), '\n model pred      ', bm.round(2), sem.round(2), ' resid sd', round(rsm,2))
```