## Step 35: baseline factual fit recorded

**What the last result showed.**
- On the 100 tuning stays, the original model's value NLL is 1.494 nats per reading over 5,424 readings. By event:

| reading | NLL | mean absolute error |
|---|---|---|
| finger glucose | 1.55 | 11.3 mg/dL |
| lab glucose | 1.24 | 9.1 mg/dL |
| arterial MAP | 1.35 | 3.9 mmHg |
| cuff MAP | 1.69 | 6.1 mmHg |

- The allowed worsening is 0.005 nats, about 0.3% of the baseline.

**What I will do next.** Inspect the per-reading CSV (columns, predicted mean, bin NLL). Check how the predicted mean relates to the bins, which I need for any local head implementation. In particular:
- which readings are included (do readings at the same time as a drug bundle count?);
- whether each row records its event time.

**What would count as evidence.** The column layout determines how I can pair per-reading predictions with states later.

```python
fa = pd.read_csv('/work/fwd_2/factual_original.csv')
print(fa.shape, fa.columns.tolist())
print(fa.head(20).to_string())
print(fa.groupby('event').size() if 'event' in fa.columns else '')
```