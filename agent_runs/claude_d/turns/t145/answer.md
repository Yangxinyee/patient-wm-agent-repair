These factual-residual regressions use only 100 stays, so they are noisy. MAP residuals rise with the preceding NE rate (+1.0 ± 0.5 arterial, +2.3 ± 0.8 cuff): the model is missing part of the NE effect even in factual prediction. That fits a model that partly absorbs NE's effect through the MAP history and should be corrected relative to what the readings alone predict. For glucose, the coefficient on the insulin surprise is 0.1 ± 1.0, too imprecise to tell whether the expected −1.5 lag-1 deficit is there.

Now that my offline head reproduction is exact, a states pass is the most useful use of the tool. From it I can compute factual predictions, `with_code` inputs for decoders and residuals. I'll draw 300 more random training stays (outside selS) and request their states in three passes. This is the first selection.

```python
rng0 = np.random.default_rng(2024)
all_tr = np.array(sorted(TRAIN.stay_id.unique())); cand = np.setdiff1d(all_tr, np.array(selS))
selT = [int(x) for x in rng0.choice(cand, 300, replace=False)]
json.dump(selT, open('/work/selT.json','w'))
print(json.dumps(selT[:100]))
```