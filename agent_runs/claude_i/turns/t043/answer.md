The stored kernels match the notebook. Insulin: about −2 mg/dL per u/h at lags 1–3h, fading by about 10h. Norepinephrine: about +3.2 mmHg per mcg/kg/min at lag 1h, fading by 5h. `TR` is a 2724-row table, probably the per-stay-hour output of the first dose_edit pass.

To compare like with like, I need the exact readout definition: which stay-hours count as "after the drug had run", and how the per-hour shifts are averaged. Next, I will inspect forward pass 1's summary and per-stay-hour file, plus `TR`. Then I will compute the kernel-implied truth on the same stay-hours. If the files list stay-hours with a flag or a time since first dose, I can reproduce the selection exactly. If not, I'll infer it from which hours appear.

```python
print(json.dumps(json.load(open('/work/fwd_1/summary.json')), indent=1)[:2500])
de1 = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(de1.shape); print(de1.head(15).to_string()); print(de1.columns.tolist())
print(TR.head(8).to_string())
print(de1.groupby('edit').size() if 'edit' in de1.columns else '')
```