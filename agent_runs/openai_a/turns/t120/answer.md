The architecture confirms that a wrong atom embedding would contaminate both the posterior and the recurrent update. The output was truncated before the earlier identification notes, so I’ll retrieve those and the retained reconstruction settings in a smaller report. This should reveal either a specific encoding error or a reason to abandon this reconstruction route.

```python
import inspect

print("=== EARLY NOTES ===")
print(Path("/work/notes.md").read_text()[:4700])

print("\n=== VOCABULARY KEYS AND CODES ===")
print("keys:", list(VOCAB.keys()))
for k in ("codes", "event_kinds", "atom_kinds", "categorical", "units"):
    if k in VOCAB:
        print(k, repr(VOCAB[k])[:900])

print("\n=== RECENT RECONSTRUCTION VARIABLES ===")
for k, v in sorted(dict(globals()).items()):
    if k.startswith(("r96_", "r101_", "r102_", "r103_")):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, pd.DataFrame):
            print(k, "DataFrame", v.shape)
        elif callable(v):
            try:
                print(k, "function", inspect.signature(v),
                      "names:", v.__code__.co_names,
                      "constants:", repr(v.__code__.co_consts)[:250])
            except Exception:
                print(k, "callable")
        else:
            print(k, repr(v)[:450])

print("\n=== CORE EMBEDDING HELPERS ===")
for k in ("rm_insulin_atoms", "rm_set", "rm_mlp", "rm_lin", "r96_empty_world"):
    f = globals().get(k)
    if callable(f):
        print(k, inspect.signature(f))
        try:
            print(inspect.getsource(f)[:1400])
        except Exception:
            print("names:", f.__code__.co_names, "constants:", repr(f.__code__.co_consts)[:500])
```
