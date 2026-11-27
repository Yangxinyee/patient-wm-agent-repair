Earlier work reconstructed the prediction heads accurately and collected a controlled numeric-dose probe. The output cut off before its held-out diagnostic, so I cannot yet judge whether a head-only correction is viable. I’ll recover that result and the reusable functions before choosing an edit.

```python
import inspect, re
text = Path("/work/notes.md").read_text()
for sec in re.split(r"(?=^## )", text, flags=re.M):
    if sec.startswith("## Run 32:") or sec.startswith("## Run 33:"):
        print(sec)

print("\nUSER FUNCTIONS:")
for key, val in list(globals().items()):
    if callable(val) and getattr(val, "__module__", "") == "__main__":
        try:
            print(key, str(inspect.signature(val)))
        except Exception:
            print(key)

print("\nREUSABLE OBJECTS:")
for key, val in list(globals().items()):
    if any(s in key.lower() for s in ["repair", "probe", "reference", "design", "draw",
                                      "latent", "benchmark", "policy", "state_meta",
                                      "head_data", "head_cfg"]):
        if isinstance(val, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(key, type(val).__name__, tuple(val.shape))
        elif isinstance(val, dict):
            print(key, "dict keys", list(val.keys())[:10])
        elif isinstance(val, (list, tuple)):
            print(key, "length", len(val))
print("\nPOLICY SAMPLE:\n", policy.head(12).to_string(index=False))
print("\nHEAD METADATA:\n", head_data.head(2).to_string(index=False))
```
