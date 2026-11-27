Within-tier dose variation is linearly decodable from the state, but only partly: cross-validated R² is 0.68 from the full state and 0.54 from `out` alone. So the model carries the dose information but barely uses it, which fits its weak response.

A head-level correction could work. Its catch is that its response to doubled or halved doses can only be checked through the tool's rollouts.

The alternative is a full local re-implementation, which would let me test dose edits offline. It needs the exact input conventions: which categorical fields each atom carries and their indices. I lost those details in the restart.

Next I will look at `vocab.json`'s structure, briefly, to see whether those mappings are written there. If the categorical vocabularies are listed, the re-implementation becomes realistic. If not, I will commit to the head-level route.

```python
for kk,vv in VOCAB.items():
    if isinstance(vv,dict):
        print(kk, "dict", len(vv), "| keys:", list(vv.keys())[:12])
    elif isinstance(vv,list):
        print(kk, "list", len(vv), "| first:", str(vv[:3])[:200])
    else:
        print(kk, type(vv).__name__, str(vv)[:200])
```