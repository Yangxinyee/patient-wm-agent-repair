Adding status row 1 makes things much worse, so measurement atoms use the zero status row. GELU now fits slightly better than ReLU, with MAE 2.4 and NLL correlation 0.89.

The replica is approximate, which may be good enough to fit a readout direction, because the final gain gets calibrated on the real model anyway. The bigger question is whether a readout-only edit can work at all. It depends on how the model's own rollout carries a glucose shift forward from hour to hour. If shifts persist strongly, a modest lag-1 tilt accumulates; if not, the needed gain may be too large for the factual budget.

So next I'll compare the time profiles per hour 13–24: the original model's effect under insulin x2 from fwd_1, and my record-based truth on the same rows.

```python
de1=pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(de1.shape, list(de1.columns)[:12])
print(de1.head(2).to_string()[:400])
```