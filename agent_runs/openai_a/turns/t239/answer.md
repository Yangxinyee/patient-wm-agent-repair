The notebook reports consistent insulin estimates, but norepinephrine estimates vary substantially with adjustment. I’ll recover the stronger insulin candidate’s construction and the identification checks. I’ll also verify that the failed run left the validated weights intact before attempting any change.

```python
print("=== IDENTIFICATION AND EDIT DESIGN ===")
for needle in [
    "## Recovery and threshold identification",
    "## Sustained insulin target and NE flexible adjustment",
    "## Protected synthetic-contrast regression",
    "## Register-protected contrast regression",
]:
    start = _notes_recovered.find(needle)
    if start >= 0:
        end = _notes_recovered.find("\n## ", start + 4)
        section = _notes_recovered[start:end if end >= 0 else None]
        print(section[:1350])

print("=== EDIT FUNCTION DEFINITIONS ===")
for name in [
    "make_selective_contrast_graft", "make_contrast_graft",
    "cache_losses", "selective_cache",
]:
    obj = globals()[name]
    print(name, inspect.signature(obj))
    try:
        print(inspect.getsource(obj)[:2600])
    except (OSError, TypeError):
        print("Names:", obj.__code__.co_names)
        print("Locals:", obj.__code__.co_varnames)

print("=== COMPACT OBJECT INVENTORY ===")
for name in [
    "r3_candidates", "r3_choice", "protected_separators",
    "fact_c", "syn_c", "stronger_screen", "_screen_fact", "_screen_syn",
    "selective_sep_w", "selective_sep_b", "audit_tuning_ids",
]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                     for k, v in list(obj.items())[:15]})
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    else:
        print(name, repr(obj)[:350])

print("=== CHECKPOINT INTEGRITY ===")
for name, tensor in stronger6_checkpoint.items():
    a = load(name)
    a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    b = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    print(name, float(np.max(np.abs(a - b))))
```
