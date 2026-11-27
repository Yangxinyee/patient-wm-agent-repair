run_cf is fast (0.05 s per stay). It returns per-bundle O (54×192) and H (54×4×192) plus the restart index, and it reuses the tool's factual states before the edit point. RT holds what heads_on_states needs per reading (bundle index, gap, last value, age). So I can generate many teacher-forced counterfactual variants cheaply.

Before writing the generator I need four small facts:
- how tilt_mean applies the tilt direction v;
- how KI/KN are indexed by lag;
- the units of `age`, so I can locate the hour of the last reading and shift it;
- which stay-hours the dose_edit readout averages over, from the fwd_1 CSV columns.

If age is in seconds and KI[1] ≈ -2, the shift bookkeeping is straightforward.

```python
peek('tilt_mean')
print('KI', np.round(KI[:4],3), 'KN', np.round(KN[:4],3))
r=RT[(RT.stay_id==2020000000)&(RT.has_last>0)].iloc[3]; print('row', r.minute, r.age, r.dt, r.event)
de1=pd.read_csv('/work/fwd_1/dose_edit_original.csv'); print(de1.columns.tolist(), de1.shape)
print(de1.head(3).to_string()[:600])
```