The kernel still has the transition reconstruction and head-feature helpers, so they can be reused. I’ll inspect their inputs and preserve the validated edit before testing synthetic dose changes. The next useful check is whether doubling or halving a dose produces a state change substantially larger than the reconstruction error.

```python
# Preserve the currently validated edit before any further experiment.
validated_edit_141 = {
    name: load(name).clone() if isinstance(load(name), torch.Tensor)
    else np.array(load(name), copy=True)
    for name in edited()
}

print("=== TRANSITION IMPLEMENTATIONS ===")
for function_name, limit in [
    ("rec_step_with_z", 2100),
    ("r131_step", 2600),
    ("rec_context", 1300),
    ("rec_layer", 1700),
]:
    print("\nFUNCTION:", function_name)
    try:
        show_function(globals()[function_name], limit=limit)
    except Exception as exc:
        print(type(exc).__name__, str(exc)[:150])

print("\n=== RECENT RECONSTRUCTION VARIABLES ===")
recent_reconstruction = [
    (name, obj) for name, obj in list(globals().items())
    if name.startswith(("rec_", "r138", "r139", "r140", "r137"))
    and not callable(obj)
]
for name, obj in recent_reconstruction[-24:]:
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "keys", list(obj.keys())[:10])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, "length", len(obj), repr(obj[:3])[:160])
    elif isinstance(obj, (str, float, int, bool)):
        print(name, repr(obj)[:160])
```
