The architecture confirms that a dedicated pathway must carry dose information through the recurrent update—not merely alter the next-reading head. The freed latent coordinates look promising, but removing the selected attention head still causes substantial reconstruction error. I’ll recover the proposed channel’s implementation and the record-based effect estimates. A usable design must preserve the head’s ordinary contribution while adding a separately controllable dose signal.

```python
print("=== RECORD-BASED EVIDENCE ===")
print(_notes197[:4300])

print("\n=== RELEVANT WEIGHT SHAPES ===")
for n in NAMES:
    if any(s in n for s in ("z_in", "z_post", "core.prelude", "core.b_in", "core.c_out")):
        print(n, SHAPES[n])

print("\n=== RETAINED CHANNEL / REPLAY OBJECTS ===")
_items198 = list(globals().items())
_matches198 = []
for n, v in _items198:
    if n.startswith("_"):
        continue
    if isinstance(v, torch.nn.Module):
        _matches198.append((n, type(v).__name__, None))
    elif any(s in n.lower() for s in
             ("gate", "candidate", "post", "replay", "step", "probe", "policy",
              "cache", "prelude", "tokens", "private", "attention", "state")):
        if isinstance(v, dict):
            description = list(v)[:10]
        elif callable(v):
            try:
                description = str(inspect.signature(v))
            except Exception:
                description = "callable"
        else:
            description = str(getattr(v, "shape", type(v).__name__))
        _matches198.append((n, type(v).__name__, description))
for item in _matches198[-75:]:
    print(*item)

print("\n=== HELPER IMPLEMENTATION ===")
for n in ("attention_features196", "array196"):
    fn = globals().get(n)
    if callable(fn):
        try:
            print(inspect.getsource(fn))
        except Exception:
            print(n, "names:", fn.__code__.co_names, "constants:", fn.__code__.co_consts)

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 197–198 — private-channel recovery\n"
            "- Architecture re-read; no active edits.\n"
            "- Two latent coordinates reconstruct their original token contribution "
            "with held RMS .00091 versus token RMS .20016.\n"
            "- Removing the best attention head still has RMS .242: it is not dispensable "
            "without further compensation and actual factual validation.\n")
```
