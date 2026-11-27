The second batch’s factual penalty is +0.00371 nats, also below the limit. I’ll preserve this candidate and recover its construction. Further strengthening is worth testing only if it improves dose responses while keeping enough factual-fit margin.

```python
import inspect
import dis

stronger6_checkpoint = {
    n: torch.as_tensor(load(n)).clone() for n in edited()
}
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Stronger selective candidate; Python 203, forwards 21/50\n"
        "- Preserved stronger6_checkpoint (hash 98f3968009).\n"
        "- GPU factual deltas: first 100 tuning +0.00321; second 100 +0.00371.\n"
        "- Paired first-100 rollout: insulin x2 -7.613, half +3.390; "
        "NE x2 -0.504, half -0.602.\n"
        "- Better insulin magnitude, still short of the record-derived target. "
        "Further candidates must retain factual-fit margin.\n"
    )

print("=== CONSTRUCTION SIGNATURES ===")
for _fn in [make_selective_contrast_graft, make_contrast_graft,
            selective_cache, cache_losses, sc_evaluate]:
    print(_fn.__name__, inspect.signature(_fn))
    print("globals:", _fn.__code__.co_names)
    print("constants:", repr(_fn.__code__.co_consts)[:600])

print("\n=== SELECTIVE CONSTRUCTION ===")
try:
    print(inspect.getsource(make_selective_contrast_graft)[:4400])
except (OSError, TypeError):
    _lines = [
        f"{i.offset:4} {i.opname:24} {i.argrepr}"
        for i in dis.get_instructions(make_selective_contrast_graft)
    ]
    print("\n".join(_lines)[:4400])

print("\n=== RELEVANT OBJECTS ===")
_relevant = set(make_selective_contrast_graft.__code__.co_names)
_relevant.update(make_contrast_graft.__code__.co_names)
_relevant.update(n for n in globals()
                 if "checkpoint" in n or n.startswith("r3_"))
for _n in sorted(_relevant):
    if _n not in globals():
        continue
    _v = globals()[_n]
    if isinstance(_v, dict):
        print(_n, "dict", len(_v), "keys", list(_v)[:10])
    elif isinstance(_v, (np.ndarray, torch.Tensor)):
        print(_n, tuple(_v.shape))
    elif isinstance(_v, (int, float, str, bool)):
        print(_n, repr(_v)[:100])

print("\n=== EARLY RECORD EVIDENCE ===")
print(Path("/work/notes.md").read_text()[:3500])
```
