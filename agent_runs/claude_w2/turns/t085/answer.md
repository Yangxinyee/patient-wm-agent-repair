The vocab shows how predicted means are formed. Glucose_finger fine bins are 10–20 mg/dL wide, with the top bin at ≥260 (centre 271). MAP_arterial tops out at ≥98 (centre 101). So predicted means saturate at high glucose. Insulin rate bins saturate above 5 U/h, and norepinephrine rate bins above 1.0, which every doubled capped rate reaches. Doubling a dose therefore enters the model mostly through the z-scores at the top end.

**Next step: test a linear surrogate of the rollout on sel1.** For each stay I start from the recorded history up to hour 12. I then iterate the fitted one-step function for hours 13–24, once with the recorded insulin and once with insulin x2 from hour 12. The policy-band terms are recomputed from the simulated glucose. I average the shift over the reached rows that have glucose_finger readings, as the tool does. I compare four versions:
- **(B)** the model's coefficients, against the tool's actual result for the original model (−2.14);
- **(A)** the records' coefficients;
- **(C)** the model's persistence with the records' dose terms;
- the structural K-based truth (about −14.1).

**How to read it.**
- If (B) lands near −2.1 with the same lag profile, the surrogate is a usable design tool.
- (C) will show whether fixing the dose terms alone is enough, or whether persistence must be raised too.

```python
T = T2[['stay_id','h','gf','gl','ins','glu_now']].copy()
T['ins'] = T['ins'].fillna(0.0)
def surrogate(coef, stays, mult=2.0, start=12, nG=3, nD=10, nP=4):
    c0 = coef[0]; ph = coef[1:1+nG]; bd = coef[1+nG:1+nG+nD]; th = coef[1+nG+nD:1+nG+nD+nP]
    out_rows = []
    for sid_ in stays:
        S_ = T[T.stay_id == sid_].sort_values('h')
        hrs_ = S_.h.values; gfv = S_.gf.values; dv = S_.ins.values; glu = S_.glu_now.values
        Gh = pd.Series(np.where(np.isnan(gfv), glu, gfv)).ffill().fillna(150.0).values
        paths = {}
        for m_ in (1.0, mult):
            G_ = Gh.copy(); d_ = dv.copy(); d_[hrs_ >= start] *= m_
            p_ = np.where(np.isnan(glu), 0.0, pfun(np.nan_to_num(glu, nan=0.0)))
            for i_, t_ in enumerate(hrs_):
                if t_ <= start: continue
                val_ = c0 + sum(ph[j_] * G_[i_-1-j_] for j_ in range(nG) if i_-1-j_ >= 0)
                val_ += sum(bd[k_] * d_[i_-1-k_] for k_ in range(nD) if i_-1-k_ >= 0)
                val_ += sum(th[k_] * p_[i_-1-k_] for k_ in range(nP) if i_-1-k_ >= 0)
                G_[i_] = val_; p_[i_] = pfun(np.array([val_]))[0] if val_ >= 150 else 0.0
            paths[m_] = G_
        for i_, t_ in enumerate(hrs_):
            if 13 <= t_ <= 24 and not np.isnan(gfv[i_]) and dv[(hrs_ >= start) & (hrs_ < t_)].sum() > 0:
                out_rows.append((sid_, t_, paths[mult][i_] - paths[1.0][i_]))
    return pd.DataFrame(out_rows, columns=['stay_id','hour','shift'])
coefT = beta; coefM = bm_
coefC = np.concatenate([coefM[:4], coefT[4:14], coefM[14:]])
res_s = {nm_: surrogate(cf_, sel1) for nm_, cf_ in [('B_model', coefM), ('A_truthAR', coefT), ('C_modelAR_trueDose', coefC)]}
deo = de[(de.edit == 'insulin_x2') & de.reached]
print('tool original x2 reached rows', len(deo), 'mean', round(deo['shift'].mean(), 2))
print('K-truth rows in r:', len(r), 'mean truth', round(r['truth'].mean(), 2), 'model shift', round(r['shift'].mean(), 2))
for nm_, df_ in res_s.items(): print(nm_, len(df_), round(df_['shift'].mean(), 2))
lagtab = pd.DataFrame({nm_: df_.assign(lag=df_.hour-12).groupby('lag')['shift'].mean() for nm_, df_ in res_s.items()})
lagtab['tool'] = deo.assign(lag=deo.hour-12).groupby('lag')['shift'].mean(); lagtab['Ktruth'] = r.groupby('lag')['truth'].mean()
print(lagtab.round(2).to_string())
```