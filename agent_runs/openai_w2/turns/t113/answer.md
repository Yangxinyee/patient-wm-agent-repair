The architecture confirms that a dose-residual correction could enter the latent posterior before the recurrent update. No weights are edited. The previous output was truncated before the helper inventory, so I’ll retrieve that inventory alone and use it to complete the reconstruction check without spending a forward pass.

```python
import inspect

_snapshot98 = list(globals().items())

print("=== FUNCTIONS ===")
for _n98, _o98 in sorted(_snapshot98):
    if inspect.isfunction(_o98) and getattr(_o98, "__module__", "") == "__main__":
        try:
            print(_n98, str(inspect.signature(_o98)))
        except Exception:
            print(_n98)

print("\n=== ARRAYS ===")
print("; ".join(
    f"{n}:{o.shape}"
    for n, o in sorted(_snapshot98)
    if not n.startswith("_") and isinstance(o, np.ndarray)
))

print("\n=== DATAFRAMES ===")
for _n98, _o98 in sorted(_snapshot98):
    if not _n98.startswith("_") and isinstance(_o98, pd.DataFrame):
        print(_n98, _o98.shape, list(_o98.columns))

print("\n=== DICTIONARIES AND LISTS ===")
print("; ".join(
    f"{n}:{type(o).__name__}[{len(o)}]"
    for n, o in sorted(_snapshot98)
    if not n.startswith("_") and isinstance(o, (dict, list, tuple))
))
print("\nCurrent edits:", edited())
```