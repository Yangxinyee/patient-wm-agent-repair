The given-only regression matches the rollout. Within factual records the model responds to a one-unit/h dose surprise by about -0.57 ± 0.31 mg/dL. The actual effect is -1.9 to -2.0, so the model carries only about 27% of the true insulin effect even within support. It has learned to expect the glucose fall after a high reading, the readings standing in for the insulin they trigger.

The repair is constrained by factual fit. Strengthening the dose response alone would shift factual predictions by about -7 mg/dL during insulin periods, far beyond the 0.005-nat tolerance. The added effect has to be the dose minus the dose the band would predict from the triggering reading. On records that term averages zero. Under an edit it adds the missing ~73% of Σ kernel × extra dose.

Building that through prelude attention and several layer norms needs a CPU replica of the model, checked against the tool's state outputs. Step one is the tokenisation: codes, how z-scores and bins are computed, and the extra tables. I'll print the vocab contents.

```python
V = VOCAB
print(list(V.keys()))
for k_ in V:
    s_ = json.dumps(V[k_])
    print('==', k_, len(s_)); print(s_[:1500])
```