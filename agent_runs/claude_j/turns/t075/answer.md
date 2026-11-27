The tensor list gives the structure of the value head. It is a two-layer MLP, `attr_heads.value_bin.0` (192×192), then an activation, then `.2` (17×192). A per-code, per-last-bin logit table (`embed_transition.value_bin`, 221×17) is added on top. So any correction to the reading predictions has to be read from the state the head sees.

The next question is whether the recent dose history can be decoded linearly from the model's state at the bundle before each glucose reading. I already have GPU states for 5,116 bundles from `fwd_3`. First I'll check what that file contains and how the tuning records are laid out, so I can line up doses with states. If simple ridge regressions recover lag-1 to lag-3 doses well out of sample, a head-only edit is feasible. If not, I'll need to work inside the core.

```python
import json
print(S.files)
print(json.load(open('/work/fwd_3/summary.json')))
print(tu.columns.tolist()); print(tu.head(12).to_string())
print(sid[:5], hrs[:20], hi_[:20], hn_[:20], na_[:20])
```