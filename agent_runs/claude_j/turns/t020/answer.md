The "leak" was a red herring. All norepinephrine starts at map_arterial 66-72 happen at hour 0, ICU admission, with no previous reading. From hour 1 on, starting is deterministic: map_arterial below 65 on the unrounded value.

Log-rate changes have autocorrelation of zero (-0.001). So the titrated rate is a multiplicative random walk: rate_t = rate_{t-1} × exp(step(MAP band) + e_t), with i.i.d. e_t of SD about 0.29, rounded to 0.01 and capped at 1.0. The innovation e_t looks like clinician noise, analogous to the insulin multiplier. Unlike insulin it persists into all later doses, which deconvolution can handle.

Before running the norepinephrine surprise regression I need two more pieces of the policy:
1. **Measurement:** is map_arterial an arterial-line attribute per stay, and does it depend on MAP? This matters for selection.
2. **Stopping:** what rule turns norepinephrine off?

A stay-level arterial line with MAP-independent measurement would make map_arterial outcomes selection-free. A stopping rule tied to low rates or high MAP would need to be handled in the deconvolution.

```python
st = P[P.h>=0].groupby('stay_id').agg(fMA=('MA', lambda x: x.notna().mean()), fMC=('MC', lambda x: x.notna().mean()), anyNE=('anyN','max'))
print("fraction of hours with MA per stay:\n", st.fMA.round(1).value_counts().sort_index().to_string())
print("fMC:\n", st.fMC.round(1).value_counts().sort_index().to_string())
print(st.groupby((st.fMA>0.5)).anyNE.mean())
# MA measurement next hour vs current MAP among A-line stays
aline = st[st.fMA>0.5].index
za = P[P.stay_id.isin(aline)&(P.h>=0)&(P.h<=23)].copy()
za['MA_meas_next']=za.groupby('stay_id').MA.shift(-1).notna().astype(int)
print("A-line stays: P(MA next) by MA_ff:", za.groupby(pd.cut(za.MA_ff,[0,55,65,75,85,200])).MA_meas_next.mean().round(3).to_dict())
zc = P[~P.stay_id.isin(aline)&(P.h>=0)&(P.h<=23)].copy()
zc['MC_meas_next']=zc.groupby('stay_id').MC.shift(-1).notna().astype(int)
print("non-A-line: P(MC next) by MC_ff:", zc.groupby(pd.cut(zc.MC_ff,[0,55,65,75,85,200])).MC_meas_next.mean().round(3).to_dict())
print("non-A-line with NE on: P(MC next) by MC_ff:", zc[zc.N.notna()].groupby(pd.cut(zc[zc.N.notna()].MC_ff,[0,55,65,75,85,200])).MC_meas_next.mean().round(3).to_dict())
# stopping rule
z = P[(P.h>=1)&(P.N0_prev>0)].copy()
z['stop'] = z.N.isna().astype(int)
print("stop by prev rate:", z.groupby(pd.cut(z.N_prev,[0,0.02,0.03,0.05,0.1,0.2,0.5,1.01])).stop.agg(['size','mean']).round(3).to_string())
```