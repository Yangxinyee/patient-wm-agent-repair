The state rows align exactly. The probe changes the output vector only slightly (RMS 0.012), despite larger internal-slot changes. A targeted readout edit remains plausible, but we still need to separate dose sensitivity from factual physiology. I’ll recover the head-building functions and remaining causal checks before fitting that edit.

```python
print("=== FUNCTIONS ===")
for k, v in sorted(list(globals().items())):
    if inspect.isfunction(v) and not k.startswith("_"):
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k)

print("\n=== REMAINING INVENTORY ===")
remaining_inventory = []
for k, v in sorted(list(globals().items())):
    if k < "hd" or k.startswith("_") or inspect.isfunction(v):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)):
        remaining_inventory.append(f"{k}:{type(v).__name__}{tuple(v.shape)}")
    elif hasattr(v, "columns") and hasattr(v, "shape"):
        remaining_inventory.append(f"{k}:{type(v).__name__}{v.shape}:{list(v.columns)[:18]}")
    elif isinstance(v, dict):
        remaining_inventory.append(f"{k}:dict({len(v)}):{list(v.keys())[:15]}")
print("\n".join(remaining_inventory)[:2800])

print("\n=== HEAD FEATURES AND CONFIG ===")
print("cfg:", cfg)
if "head_original_features" in globals():
    print("head_original_features:", type(head_original_features))
    if isinstance(head_original_features, dict):
        print({k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
               for k, v in head_original_features.items()})
    elif hasattr(head_original_features, "shape"):
        print(head_original_features.shape)
print("DML stored objects:",
      {k: {kk: (getattr(vv, "shape", None) or type(vv).__name__)
           for kk, vv in v.items()}
       for k, v in dml_models.items()})

print("\n=== NOTEBOOK HEADINGS ===")
print("Length:", len(notes_recovered))
print("\n".join(line for line in notes_recovered.splitlines() if line.startswith("##")))

print("\n=== REMAINING EARLY EVIDENCE ===")
start_early = notes_recovered.find("## Run 6")
print(notes_recovered[start_early:start_early+4500] if start_early >= 0
      else notes_recovered[7800:12300])
```