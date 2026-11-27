The architecture confirms that edited doses affect predictions through the recurrent state, not the last-reading registers. Before designing a replacement pathway, I’ll recover the saved analysis objects and exact stay IDs. The next paired factual test will tell us whether removing action information is harmless or too costly.

```python
import os, json
print("Tuning IDs:", tuning_ids_repair[:100])
print("Forward files:")
for j in range(1, 7):
    print(j, os.listdir(f"/work/fwd_{j}"))
print("Arrays/dataframes:")
inventory_repair = []
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"In", "Out"}:
        continue
    if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        inventory_repair.append((k, type(v).__name__, list(v.shape)))
print(json.dumps(inventory_repair))
print("Functions:", [k for k, v in list(globals().items())
                     if callable(v) and getattr(v, "__module__", "") == "__main__"])
print("Dictionaries:", [(k, len(v), list(v)[:6]) for k, v in list(globals().items())
                        if isinstance(v, dict) and not k.startswith("_")
                        and k not in {"SHAPES", "VOCAB", "In", "Out"}])
print("Config:", VOCAB["config"])
print("Head/action tensor shapes:",
      [(n, SHAPES[n]) for n in NAMES
       if n.startswith(("attr_heads.value_bin", "encode_action", "z_head", "z_in"))])
```
