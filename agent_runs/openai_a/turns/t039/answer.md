The architecture confirms that predictions use both the recurrent state and a last-value register. The output was truncated before the retained variables, so I still cannot check the suspected mismatch. I’ll inspect those variables and the head’s input layout directly; an incorrect register layout or normalization would support the hypothesis.

```python
import dis, re
print("=== NOTEBOOK INDEX AND START ===")
print("\n".join(re.findall(r"^##.*$", notes_text, flags=re.M)))
print(notes_text[:3200])
print("\n=== RETAINED OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"In", "Out", "NAMES", "SHAPES", "VOCAB", "notes_text"}:
        continue
    if isinstance(obj, pd.DataFrame):
        print(name, "DF", obj.shape, list(obj.columns)[:14])
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, types.FunctionType):
        print(name, str(inspect.signature(obj)))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), list(obj)[:8])
print("\n=== VOCAB KEYS AND NORMALIZATION ===")
print(list(VOCAB))
for key in VOCAB:
    if key not in {"value_bin_edges", "value_bin_centres", "rate_bin_edges",
                   "rate_bin_centres", "dose_bin_edges", "dose_bin_centres"}:
        text = str(VOCAB[key])
        print(key, text[:1600])
print("\n=== HEAD-RELATED SHAPES ===")
for name in NAMES:
    if name.startswith(("last_", "query_", "type_", "attr_heads.value", "embed.numeric",
                        "core.z_head", "core.z_prior", "core.gap", "core.out_norm")):
        print(name, SHAPES[name])
```
