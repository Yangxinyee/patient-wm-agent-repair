The cost is not concentrated in a few extreme rows. About half comes from glucose_finger rows where the edit shifts the prediction by 2–4 mg/dL, and rows with shifts above 8 account for only about 7%. On recorded data the edit's shifts explain little of the original residuals (slope 0.16). That is the known price of the lag-1 overshoot the rollouts need, so capping would not buy headroom.

The choice is therefore −2.6 (closer insulin readouts) against −2.5 (lower risk). The deciding number is the population mean cost at −2.6, currently 0.0031 ± about 0.00075 from only four batches of 100 stays.

I will tighten that estimate with factual runs on training stays that were never used to fit the readout. Training stays are in-sample for the base model, which if anything should make the cost estimate conservative. If the pooled mean stays near or below 0.003, I keep −2.6. If it is about 0.0034 or higher, I go back to −2.5. First I draw and save four batches of 100 unused training stays.

```python
used=set()
for k_,v_ in lists.items():
    try: used|=set(int(x) for x in v_)
    except Exception: pass
used|=set(int(x) for x in np.unique(A_.stay.values))
pool=[int(s) for s in stays_tr if int(s) not in used]
rng=np.random.default_rng(2024); pick=rng.choice(pool,size=400,replace=False)
batches=[sorted(int(x) for x in pick[i*100:(i+1)*100]) for i in range(4)]
json.dump(batches,open('/work/sel_cost.json','w'))
print(len(used),len(pool)); print(json.dumps(batches[0]))
```