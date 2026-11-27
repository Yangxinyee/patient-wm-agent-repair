The factual pass returned 5,306 readings across 100 training stays, with overall value NLL 1.481. By event:

| event | NLL | mean abs error |
|---|---|---|
| finger glucose | 1.449 | 11.0 |
| lab glucose | 1.239 | 9.3 |
| arterial MAP | 1.381 | 3.9 |
| cuff MAP | 1.843 | 5.9 |

The NLL is over 16 fine bins, which explains these magnitudes.

Next I'll examine the CSV layout and merge the model's predicted means into my hourly panel. Then I'll measure its learned one-step response, using the same within-band random-rate design as for the records. For the next finger glucose after an insulin dose, I'll regress the model's prediction on the dose controlling for the decision reading. For the next arterial MAP after a norepinephrine change, the same idea. The predictions are nearly noise-free, so even 100 stays should give a sharp estimate.

```python
fa = pd.read_csv('/work/fwd_2/factual_original.csv')
print(fa.shape, fa.columns.tolist()); print(fa.head(10).to_string())
print(fa.event.value_counts() if 'event' in fa.columns else '')
```