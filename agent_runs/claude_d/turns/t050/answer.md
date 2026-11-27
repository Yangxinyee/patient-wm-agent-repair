NE behaves differently from insulin. Per unit of rate, the lag-1 MAP effect falls steeply as the rate rises (10.8, 7.3, 5.7, 4.6 mmHg per mcg/kg/min across rate quartiles). Per unit of log-rate, it is roughly constant (2.7, 3.5, 3.5, 3.1 mmHg). So NE acts approximately on log-rate (or through a saturating Emax that looks the same over 0.1–1).

That makes my earlier gam_ne, which is linear in rate, misspecified. Under a log model, ×2 and ×½ should give shifts of about equal size and opposite sign. I'll re-estimate the NE effect with a log-rate blip model.

First I want to see exactly which stay-hours the dose_edit readout averages over, so my "true" targets are computed on the same rows. I'll inspect fwd_1's CSV (columns and a few rows).

```python
de = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(de.columns.tolist(), de.shape)
print(de.head(12).to_string())
print(de.groupby('edit').size() if 'edit' in de.columns else '')
```