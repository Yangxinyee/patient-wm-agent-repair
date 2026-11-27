The rollout improved insulin responses to −5.993 and +3.126, but they remain weaker than the record-derived targets. Norepinephrine responses barely changed. I will preserve this validated checkpoint and inspect the edit construction to seek a stronger correction within the remaining factual-loss margin. A stronger edit must improve rollout effects without breaching 0.005 nats.

```python
import inspect, dis
from pathlib import Path

finger_scale3_checkpoint = {n: load(n).clone() for n in edited()}
finger_scale3_validation = {
    "factual_delta_first100": 1.63755 - 1.63461,
    "insulin_x2": -5.993, "insulin_half": 3.126,
    "ne_x2": -0.514, "ne_half": -0.611,
}

for _fname in ("make_selective_contrast_graft", "cache_losses"):
    _fn = globals()[_fname]
    print("\nFUNCTION", _fname, inspect.signature(_fn))
    try:
        print(inspect.getsource(_fn))
    except Exception:
        print("LOCALS:", _fn.__code__.co_varnames)
        print("GLOBALS:", _fn.__code__.co_names)
        _ops = [f"{i.opname} {i.argrepr}" for i in dis.get_instructions(_fn)]
        print("\n".join(" | ".join(_ops[j:j+4])
                        for j in range(0, len(_ops), 4)))

print("\nTRIAL PARAMETERS:")
for _name in ("trial_r3", "protected_separators", "r3_candidates"):
    _obj = globals().get(_name)
    if isinstance(_obj, dict):
        print(_name)
        for _k, _v in list(_obj.items())[:5]:
            if isinstance(_v, dict):
                print(_k, {a: (getattr(b, "shape", None)
                              if hasattr(b, "shape") else b)
                           for a, b in _v.items()})
            else:
                print(_k, type(_v).__name__, getattr(_v, "shape", None))

_text = Path("/work/notes.md").read_text()
_start = _text.find("## Recovery and threshold identification")
_end = _text.find("\n## ", _start + 4)
print("\nTHRESHOLD EVIDENCE:")
print(_text[_start:_end][:2400])

with Path("/work/notes.md").open("a") as _f:
    _f.write(
        "\n## Finger-selective GPU validation; Python 199, forwards 18/50\n"
        "- Hash 46c1a1e52a: first-100 tuning factual delta +0.00294 nats.\n"
        "- Insulin x2 -5.993, half +3.126; NE x2 -0.514, half -0.611.\n"
        "- Preserved finger_scale3_checkpoint; stronger than scale2, "
        "but still below record-derived insulin magnitude.\n"
        "- Broader factual validation remains required.\n"
    )
```