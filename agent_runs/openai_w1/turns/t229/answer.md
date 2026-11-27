The recovered record analyses support negative insulin effects and positive norepinephrine effects after adjustment for pre-dose history, though that identification remains conditional on the adjustment being adequate. The architecture confirms that doses enter through a separate action encoder. I’ll recover the tested numeric edits and exact attention helpers next; this will distinguish an already-useful action route from a latent route that needs a relay.

```python
import re, inspect, numpy as np, torch
from pathlib import Path

secs204 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
print("=== NUMERIC-DOSE EXPERIMENTS ===")
hits204 = [s for s in secs204 if re.search(r"numeric|5x", s, re.I)]
for s in hits204:
    h = s.splitlines()[0] if s.splitlines() else ""
    nums = [int(x) for x in re.findall(r"\b(?:Run|Runs)\s+(\d+)", h)]
    if nums and nums[0] < 173:
        print(s[:850])
        if sum(len(x[:850]) for x in hits204[:hits204.index(s)+1]) > 5000:
            break

print("\n=== RELEVANT TENSORS ===")
for n in NAMES:
    if (n.startswith(("embed.numeric", "core.z_in", "core.prelude.0.attn",
                      "core.b_in", "core.c_out", "core.coda"))
        or n.startswith("encode_action.")):
        print(n, SHAPES[n])

print("\n=== CORE HELPER NAMES ===")
for n in sorted(list(globals())):
    o = globals()[n]
    if callable(o) and n.startswith(("rx_", "rr_", "rp_")):
        print(n, getattr(getattr(o, "__code__", None), "co_varnames", ())[:10])

print("\n=== CARRIER / NUMERIC CACHE KEYS ===")
for n in ("carrier201_saved", "rp_fallback_numeric", "rx_weights", "rx_layout"):
    if n in globals():
        o = globals()[n]
        if isinstance(o, dict):
            print(n, list(o.keys()))
        elif isinstance(o, (np.ndarray, torch.Tensor)):
            print(n, tuple(o.shape))
        else:
            print(n, type(o).__name__, str(o)[:100])
for n in sorted(list(globals())):
    if re.search(r"carrier|gate|compens|latent|zinput|zin|posterior", n, re.I):
        o = globals()[n]
        if isinstance(o, (np.ndarray, torch.Tensor)):
            print(n, tuple(o.shape))
        elif isinstance(o, dict):
            print(n, "keys", list(o.keys())[:6])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 204\nRecord diagnostics support opposing drug effects, conditional on pre-dose adjustment. Recovering numeric-edit evidence and exact attention helpers before designing a relay. No weights changed. Counts: 204 Python runs, 18 forwards.\n")
```