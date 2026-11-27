The adjusted estimates are consistent with an immediate glucose decrease of about 2 per insulin rate unit and a MAP increase of about 3.3 per norepinephrine rate unit. Those are observational estimates, not proof of the full rollout response. I’ll recover the longer-lag checks and head reconstruction details; a head edit remains viable only if its features behave reliably across individual latent draws.

```python
print("=== LATER CAUSAL AND ROLLOUT NOTES ===")
s = notes_text.find("## Run 6")
e = notes_text.find("## Run 19", s)
print(notes_text[s:e if e > s else s + 4500][:4500])

print("\n=== VOCAB LAYOUT ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))

print("\n=== HEAD HELPER DEFINITIONS ===")
for name in ("rh_eval", "rh_register", "wlin", "wmlp", "wnorm"):
    f = globals().get(name)
    if f is None:
        continue
    print("\n", name, inspect.signature(f))
    try:
        print(inspect.getsource(f))
    except Exception:
        print("locals:", f.__code__.co_varnames)
        print("constants:", repr(f.__code__.co_consts)[:1800])

print("\n=== RECONSTRUCTION CONFIGURATIONS ===")
for name, obj in list(globals().items()):
    if name.startswith(("wr_", "rec_", "mc_", "repair_")):
        if isinstance(obj, dict):
            print(name, {
                k: (tuple(v.shape) if hasattr(v, "shape") else
                    list(v)[:12] if isinstance(v, dict) else repr(v)[:250])
                for k, v in obj.items()
            })
        elif isinstance(obj, (str, int, float, tuple, list)) and not callable(obj):
            print(name, repr(obj)[:700])
print("\n=== QUERY/GATE RELATED GLOBAL NAMES ===")
print([n for n in globals() if any(
    x in n.lower() for x in ("query", "project", "gate", "latent", "recon")
)])
```
