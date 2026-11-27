The R99 table maps each of the 5,521 readings to its stay, hour, event and value. My recomputed logits already include the transition bias. My reconstructed mean NLL is 1.5507, while the tool reported 1.520 for these stays (fwd_4). So the reconstruction is either off by about 0.03 nats, or measured differently, for example over a different set of readings or the 4-draw z averaging.

I'll match my per-reading values to the tool's per-reading CSV from fwd_4, checking NLL and predicted mean. A high correlation with a small constant offset means my recon is good enough to cost edits by differences. Large scattered errors mean the z draws or a missing term matter, and I'd need to account for them before trusting any offline costing.

```python
import pandas as pd
f4 = pd.read_csv('/work/fwd_4/factual_original.csv')
print(f4.shape, list(f4.columns))
print(f4.iloc[0].to_dict())
vc = VOCAB['value_bin_centres']; print(type(vc), (list(vc.keys())[:5] if isinstance(vc, dict) else len(vc)))
```