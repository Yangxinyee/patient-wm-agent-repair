The original weights are restored. The cached paired states offer a way to learn which hidden-state changes carry dose information and connect them directly to value predictions. I’ll check the head reconstruction and prior attempts first; held-out decoding and factual fit will determine whether that route is usable.

```python
import inspect, re
from pathlib import Path

for name184, limit184 in [
    ("ce_make_heads", 2600), ("rx_layer2", 1900),
    ("rx_attn", 1100), ("rr_mlp", 900), ("rp_install", 1500)
]:
    fn184 = globals().get(name184)
    print("\nFUNCTION", name184)
    try:
        print(inspect.getsource(fn184)[:limit184])
    except Exception:
        print("names", fn184.__code__.co_names)
        print("constants", str(fn184.__code__.co_consts)[:limit184])

print("\nHEAD / PAIRED-STATE CACHES")
desc184 = []
for k, v in list(globals().items()):
    if k.startswith(("he_", "hp_", "r165_", "r180_", "rx_")) and not callable(v):
        if hasattr(v, "shape"):
            detail184 = str(tuple(v.shape))
        elif isinstance(v, dict):
            detail184 = "keys=" + str(list(v)[:8])
        else:
            detail184 = type(v).__name__
        desc184.append(k + ": " + detail184)
print("; ".join(desc184)[:1800])

text184 = Path("/work/notes.md").read_text()
sections184 = re.split(r"(?m)(?=^## )", text184)
chosen184 = []
for section184 in sections184:
    heading184 = section184.splitlines()[0] if section184 else ""
    m184 = re.search(r"Run(?:s)? (\d+)", heading184)
    if m184 and 18 <= int(m184.group(1)) <= 148:
        body184 = " ".join(section184.splitlines()[1:])
        if any(s in body184.lower() for s in ["edit", "exact", "nll", "head"]):
            chosen184.append(heading184 + ": " + body184[:190])
print("\nPRIOR APPROACHES\n" + "\n".join(chosen184)[:2400])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 184\nInvestigating direct value-head correction using existing dose-sensitive "
            "state contrasts, with factual-zero anchoring and stay-held-out validation. "
            "Recovering only necessary CPU helpers and prior approach summaries. No edits.\n")
```