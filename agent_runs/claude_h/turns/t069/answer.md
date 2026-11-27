The architecture settles the remaining worry. Drug doses reach the model in only two places:
- **Atom embeddings.** The dose and rate bins and the `z_dose`/`z_rate` numeric columns feed `encode_action` and then u_k, which drives the state. My edit neutralises this path completely.
- **Per-code "last" register.** `z_last_rate` and the last rate bin hold the most recent earlier value of the same code. For glucose and MAP codes these fields are always absent, because readings have no rate. They only affect predictions of drug attributes, which neither the rollout (doses are given) nor the value NLL uses.

So the edited model's reading predictions cannot depend on dose values in any stay. That matches pass 3's exact zeros. The factual fit is unchanged on two separate sets of 100 tuning stays (+0.00005 and −0.00007 nats, against a 0.005 allowance).

Since the records support absent effects for both drugs and the edit delivers exactly that, I'll make a last check that the stored edit is exactly as intended and record the final conclusion in the notebook. The edit must be bin rows 1–16 equal to the original row 8, row 0 untouched, and only the `z_dose` and `z_rate` numeric columns zeroed. Any other difference would mean the saved tensors don't match what I tested.

```python
import numpy as np
print('edited:', edited())
for n in ['embed.bins.dose.weight','embed.bins.rate.weight']:
    a=np.asarray(load(n)); o=np.asarray(load_original(n))
    print(n, a.shape, 'row0 dev', float(np.abs(a[0]-o[0]).max()),
          'rows1..16 vs orig row8 dev', float(np.abs(a[1:17]-o[8][None]).max()))
a=np.asarray(load('embed.numeric.weight')); o=np.asarray(load_original('embed.numeric.weight'))
print('numeric', a.shape, 'cols2,4 absmax', float(np.abs(a[:,[2,4]]).max()),
      'other cols dev', float(np.abs(a[:,[0,1,3,5]]-o[:,[0,1,3,5]]).max()))
with open('/work/notes.md','a') as f:
    f.write("""
## Final (runs 64-65)
- ARCHITECTURE: dose values enter only via drug-atom embeddings (bins.dose, bins.rate, numeric z_dose/z_rate) -> encode_action -> u_k,
  and via the per-code 'last' register (z_last_rate, last rate bin) which for glucose/MAP codes never holds a rate (only drug-attribute
  heads use it; not used in rollout or value NLL). Edit neutralizes path 1 => reading predictions exactly dose-invariant on any stay.
- Final belief: insulin effect on glucose ABSENT (jitter LP ~0, RD at sliding-scale thresholds ~0, implied x2 +0.9+-0.5);
  NE effect on MAP ABSENT or negligible (implied x2 -0.34+-0.14, spec-fragile, t>=12 ~0); policies depend on recorded readings
  only (sharp thresholds, no art/cuff dilution) so jitter identification is clean. Expected true readouts ~0 for all four.
- SUBMIT dose-invariant edit: readouts exactly 0 (FWD 3), NLL unchanged (FWD 2: +0.00005, FWD 4: -0.00007).
""")
print('notes updated')
```