Replacement mixtures look less costly than additive ones, but the reported losses need comparison with the same baseline. I’ll calculate those differences and check rollout behavior on another group of stays before choosing a trial. That will distinguish a broadly useful correction from a first-100-specific result.

```python
import numpy as np
import pandas as pd
import json

rows272 = []
for a272, b272 in [(1.0, 0.0), (0.95, 0.05), (0.90, 0.10),
                   (0.85, 0.15), (0.80, 0.20), (0.95, 0.0),
                   (1.05, 0.0), (1.0625, 0.0)]:
    row272 = {"gated": a272, "ungated": b272}
    for split272 in ["first", "extra", "tuning"]:
        base272 = blend_eval268(blend_factparts268[split272], 1.0, 0.0)
        rr272 = blend_eval268(blend_factparts268[split272], a272, b272)
        row272[split272 + "_NLL_delta"] = rr272[0] - base272[0]
    m1272 = blend_eval268(blend_synparts268[1.0], a272, b272)[2]
    row272["local_x2"] = blend_eval268(
        blend_synparts268[2.0], a272, b272)[2] - m1272
    row272["local_half"] = blend_eval268(
        blend_synparts268[0.5], a272, b272)[2] - m1272
    rows272.append(row272)
print(pd.DataFrame(rows272).to_string(index=False))

probe272 = make_blend268(1.0, 0.0)
print("\nBlend reconstruction maximum differences:",
      {n: float(np.max(np.abs(np.asarray(probe272[n]) -
                             np.asarray(validated_gated8_checkpoint[n]))))
       for n in validated_gated8_checkpoint})
print("Current weights still equal validated gated8:",
      all(np.array_equal(np.asarray(load(n)),
                         np.asarray(validated_gated8_checkpoint[n]))
          for n in validated_gated8_checkpoint))

head272 = pd.read_csv("/data/tuning.csv", nrows=1)
idcol272 = next(c for c in head272.columns if "stay" in c.lower())
tuning_ids272 = pd.read_csv("/data/tuning.csv", usecols=[idcol272])[idcol272].unique()
print("\nLAST100_TUNING_IDS:",
      json.dumps([int(x) for x in tuning_ids272[-100:]]))
```
