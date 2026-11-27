Helper signatures recovered:
- `panel(df)` builds the hourly panel.
- `base_ins(g)` is the insulin base-rate policy.
- `true_shift(...)` computes my record-based truth.
- There is also a partial numpy reimplementation of the embeddings and encoders (`atom_vec`, `encode_set`, `mlp`, `LN`).

The value head (17 bins) and the state layout are clear. Next I'll save a short note so a restart doesn't lose the plan. Then I'll check the hourly panel and the states file, so I can join each bundle's state to its recorded dose and reading.

```python
with open('/work/notes.md','a') as fh:
    fh.write("""
## Runs 30-41 (after restart)
- fwd_2 = factual ORIGINAL on tuning setA: value NLL/reading 1.51903 (finger 1.531, lab 1.261, art 1.349, cuff 1.746). Baseline for +0.005 limit.
- fwd_3 = states ORIGINAL on tuning setA (5075 bundle rows; layout out|h0..h3|zpost).
- Runs 30-37 tried to reimplement core locally: not matching yet (best err 1.79 on z after bundle0). Parked.
- Value head = MLP 192->192->17 (+ embed_transition[code*17+lastbin]); shared by all 4 value codes.
- PLAN: readout edit. Decode a dose-innovation feature (dose minus policy-expected dose given reading, lagged) from head input
  computed from states; repurpose one hidden unit of attr_heads.value_bin per drug to add a logit tilt (gated to glucose / MAP codes);
  calibrate gain with dose_edit on setA vs my truth (-14.17/+7.09/+4.82/-2.41). Check factual with 'both'.
""")
print(Ptu.shape, list(Ptu.columns)[:30])
print(Ptu.head(3).to_string()[:900])
print({k:z[k].shape for k in z.files})
import numpy as np
print(np.unique(np.round((z['hour']%1),3))[:20])
```