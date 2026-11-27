Batch 3 is in (fwd_9). I now have five sets of original-model states on train stays (fwd_5 to fwd_9, about 500 stays). Before spending passes on batches 4–5, I'll check whether more data actually helps.

For every reading in these stays, this run computes:
- the local head's predicted mean and NLL;
- the head input `with_code`, averaged over z draws;
- the lag-1 to lag-3 insulin dose deviations from the records.

Then I fit a ridge read-out of the lag-1 deviation on glucose-finger rows from 1, 3 and 5 files. I compare stay-grouped cross-validated R² as the training set grows, and re-estimate the model's and the truth's one-step coefficients on all rows.

If R² rises clearly with data (say from 0.25 to 0.35 or more), I should collect more batches. If it plateaus, the head simply cannot see more of the dose, and I proceed with what I have.

```python
import pickle
files = [f'/work/fwd_{k}/states_original.npz' for k in [5,6,7,8,9]]
ALLR, ALLW = [], []
for fi, fn in enumerate(files):
    Xs_, R_ = build_reading_rows(fn, Gtr)
    Xs_ = Xs_ if torch.is_tensor(Xs_) else torch.as_tensor(np.asarray(Xs_))
    pm_, nll_, P_ = head_eval(Xs_, R_, nz=16)
    R_['pm'] = pm_; R_['nll'] = nll_; R_['file'] = fi
    Q8 = q_from_rows(Xs_, torch.as_tensor(R_.prev.values.astype(int)), torch.as_tensor(R_.dt.values, dtype=torch.float64), nz=8, seed=11)
    f_ = reg_feats(torch.as_tensor(R_.has.values), torch.as_tensor(R_.age.values), torch.as_tensor(R_.zl.values))
    const = T('embed.code.weight')[torch.as_tensor(R_.code.values.astype(int))] + T('embed.bins.value.weight')[torch.as_tensor(R_.lvb.values.astype(int))] + f_ @ T('last_numeric.weight').T + T('last_numeric.bias')
    ALLR.append(R_); ALLW.append((Q8.mean(0) + const).numpy())
RA_ = pd.concat(ALLR, ignore_index=True); WA_ = np.concatenate(ALLW)
hr = RA_.hour.round().astype(int).values
for k in [1,2,3]:
    RA_[f'dev{k}'] = [DEV.loc[s, h-k] if (h-k) in DEV.columns else 0.0 for s, h in zip(RA_.stay.values, hr)]
    RA_[f'dev{k}'] = RA_[f'dev{k}'].fillna(0.0)
pickle.dump((RA_, WA_), open('/work/readrows_tr.pkl','wb'))
print(RA_.groupby('event').size().to_dict(), 'local NLL by event', RA_.groupby('event').nll.mean().round(4).to_dict())
mg = (RA_.event=='glucose_finger').values
Rg, Wg = RA_[mg].reset_index(drop=True), WA_[mg]
lvb_oh_g = np.eye(17)[Rg.lvb.values.astype(int)]
for nf in [1, 3, 5]:
    sub = Rg.file.values < nf
    F_ = np.c_[Wg[sub], lvb_oh_g[sub]]; y = Rg.dev1.values[sub]
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2,4,14)), F_, y, cv=GroupKFold(5), groups=Rg.stay.values[sub])
    print(f'files<{nf}: rows {sub.sum()}  dev1 CV R2 {1-np.mean((y-pred)**2)/y.var():.3f}   (dev1 sd {y.std():.3f})')
# one-step coefficient comparison on all 500 stays
Ds = []
for fi, R_ in enumerate(ALLR):
    sts = sorted(set(int(s) for s in R_.stay.values))
    D_ = build_ins_design2(Gtr, sts); D_['sid'] = np.array(sts)[D_.s.values.astype(int)]; D_['hour'] = D_.hour.round().astype(int)
    gg = R_[R_.event=='glucose_finger'][['stay','hour','pm']].rename(columns={'stay':'sid'}); gg['hour'] = gg.hour.round().astype(int)
    Ds.append(D_.merge(gg, on=['sid','hour'], how='left'))
DS = pd.concat(Ds, ignore_index=True); DS = DS[DS.pm.notna()].reset_index(drop=True)
b_y, s_y, _ = my_reg(DS, 'y'); b_m, s_m, rm2 = my_reg(DS, 'pm')
print('rows', len(DS), '\ntruth dev:', b_y[dv].round(2).values, ' se', s_y[dv].round(2).values, '\nmodel dev:', b_m[dv].round(2).values, ' se', s_m[dv].round(2).values)
```