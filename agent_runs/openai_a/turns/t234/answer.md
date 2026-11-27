The saved evidence supports an insulin effect near −1.93 per dose unit with persistent glucose dynamics. Norepinephrine estimates vary with the adjustment method, so I won’t treat those associations as established effects. Next I’ll inspect the contrast fits and cached losses to find a stronger insulin candidate with acceptable factual cost.

```python
def _describe_small(v, depth=0):
    if isinstance(v, (np.ndarray, torch.Tensor)):
        return f"{type(v).__name__}{tuple(v.shape)}"
    if isinstance(v, dict):
        if depth >= 1:
            return f"dict({len(v)}), keys={list(v)[:8]}"
        return {str(k): _describe_small(x, 1)
                for k, x in list(v.items())[:8]}
    if isinstance(v, (list, tuple)):
        return [_describe_small(x, depth + 1) for x in v[:6]]
    if isinstance(v, (str, int, float, bool, np.number)):
        return repr(v)[:140]
    return type(v).__name__

print("=== CONTRAST AND CACHE OBJECTS ===")
for _n, _v in list(globals().items()):
    if _n.startswith(("sc_", "r2_", "r3_", "selective_", "finger_")):
        print(_n, _describe_small(_v))

print("\n=== R3 FIT CONTENTS ===")
for _key in [(0.1, 1.0), (0.3, 1.0)]:
    if _key in r3_candidates:
        print(_key, _describe_small(r3_candidates[_key]))
        if isinstance(r3_candidates[_key], dict):
            for _n, _v in r3_candidates[_key].items():
                print(" ", _n, _describe_small(_v))

print("\n=== CACHE HELPER ===")
print(inspect.signature(sc_cache_inputs))
print("globals:", sc_cache_inputs.__code__.co_names)
print("cache_losses bytecode:")
print("\n".join(
    f"{i.offset:4} {i.opname:22} {i.argrepr}"
    for i in dis.get_instructions(cache_losses)
)[:2600])

print("\n=== IDENTIFICATION NOTEBOOK SECTIONS ===")
_sections = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
_chosen = [
    s for s in _sections
    if re.search(r"identification|regression discontinuity|dynamic RD|NE identification|flexible",
                 s.splitlines()[0], re.I)
]
for _section in _chosen[:5]:
    print(_section[:600])
```
