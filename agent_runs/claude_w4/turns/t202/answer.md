The block responds very weakly to the prelude output. Reproducing one mg/dL of the model's natural glucose-belief change (nuG) takes a δ of norm 367, about 40% of the slot norm. So that target is infeasible through E. A reading's information probably reaches the state mainly through the memory cross-attention, not E.

The question is then whether E can at least move the next glucose prediction efficiently, along whatever direction the heads read, even if it does not mimic a reading. I'll fit a linear readout of the original model's predicted next finger mean (fwd_2) from the mean over slots of the preceding state (fwd_3), plus log gap. Then I'll combine it with the Jacobian to get the largest prediction shift per unit norm of δ. A shift of about 0.01 mg/dL or more per unit would make the E route usable; much less would rule it out.

```python
# state rows: X3 = [out(192), h slots(768), z(32)] after each bundle in df_s order
hv = X3[:, 192:960].reshape(-1,4,192).mean(1)
fg = fa[fa.event=='glucose_finger'][['stay_id','hour','predicted_mean','observed']].reset_index(drop=True)
ds_ = df_s.reset_index(drop=True).copy(); ds_['row'] = np.arange(len(ds_))
qq = fg.copy(); qq['hq'] = qq.hour - 1e-6
mm = pd.merge_asof(qq.sort_values('hq'), ds_[['stay_id','hour','row']].rename(columns={'hour':'hb'}).sort_values('hb'), left_on='hq', right_on='hb', by='stay_id', direction='backward')
mm = mm.dropna(subset=['row'])
rows_ = mm.row.astype(int).values
Xr = np.c_[hv[rows_], np.log(np.maximum((mm.hour - mm.hb).values*3600,1.0))]
yr = mm.predicted_mean.values
grp_ = mm.stay_id.values
print('n', len(yr), 'CV R2 pred-mean from mean-slots(y): %.3f' % cvr2(Xr, yr, grp_, alpha=1.0))
rg = Ridge(alpha=1.0).fit(Xr, yr); w_m = rg.coef_[:192]
Jm_ = J.reshape(len(selJ),4,192,192).mean(1)          # d mean_slots(y) / d delta
g_d = np.einsum('i,nij->nj', w_m, Jm_).mean(0)
print('|w_m| %.4f  max pred shift per unit |delta| %.5f mg/dL' % (np.linalg.norm(w_m), np.linalg.norm(g_d)))
print('nuG-based delta pred shift per unit |delta| %.5f' % (0.554/367.13))
```