The factual pass worked. On these 100 training stays the original model's value NLL is 1.436 nats per reading over 5673 readings: glucose_finger 1.538, glucose_lab 1.304, map_arterial 1.331, map_cuff 1.597. This is the baseline the edit must stay within 0.005 of on the evaluation stays.

Next I'll look at the per-reading file and implement the value head locally. The head is the view, the query norm with a type embedding, the last-value register, the value MLP and the transition table. I'll average over prior z samples. Then I'll compare its predicted means and bin NLLs with the file, testing the undocumented choices (type index, last_numeric layout, last_status index). The right choices should give near-exact agreement; any leftover spread should be small, coming only from the 4 random prior samples.

```python
fac3 = pd.read_csv('/work/fwd_3/factual_original.csv')
print(fac3.columns.tolist()); print(fac3.head(8).to_string()); print(len(fac3))
print(fac3.describe().T.to_string()[:1500])
```