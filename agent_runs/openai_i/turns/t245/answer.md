The records support effects near −2 glucose units per insulin-rate unit and +3 MAP units per norepinephrine-rate unit, subject to observational identification assumptions. The output was truncated before the routing helpers appeared. I’ll retrieve those compactly so I can test whether a closed-form feed-forward correction preserves the factual behavior lost by explicit routing.

```python
import re, textwrap, inspect
from pathlib import Path

wanted219 = sorted(
    n for n in list(globals())
    if not n.startswith("_") and
    (any(n.endswith(str(k)) for k in
         (196, 201, 203, 204, 208, 209, 214, 215, 216, 217))
     or any(w in n.lower() for w in
            ("replay", "policy", "proximal", "persistence", "w136", "tuning_ids")))
)
print("CACHED OBJECT NAMES")
print(textwrap.fill(", ".join(wanted219), width=125))

print("\nHELPER CONTRACTS")
for n in ("block136", "attention_features196", "linear136", "norm136",
          "replay136", "step136", "forward136", "token_response214",
          "anchored_readout214", "private_posterior208"):
    obj = globals().get(n)
    if callable(obj):
        print(n, inspect.signature(obj))
        print("  globals:", obj.__code__.co_names)
for n in wanted219:
    obj = globals()[n]
    if callable(obj) and any(w in n.lower() for w in
                            ("replay", "response", "readout", "prelude", "private")):
        print(n, inspect.signature(obj), "globals:", obj.__code__.co_names)

print("\nKEY CACHE SIZES")
for n in ("input196", "old_tokens196", "old_attention_features196", "all_z201",
          "source201", "train196", "H201", "HF201", "target_att201",
          "private_basis196", "anchor196", "axis_ins196", "axis_ne196",
          "W136", "w136", "neutral209", "tuning_ids"):
    obj = globals().get(n)
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(n, tuple(obj.shape), str(obj.dtype))
    elif isinstance(obj, dict):
        print(n, "dict", len(obj), "first keys", list(obj)[:8])
    elif obj is not None:
        print(n, repr(obj)[:180])

print("\nRECORD-BASED DYNAMICS NOTES")
budget219 = 2700
for line in Path("/work/notes.md").read_text().splitlines():
    if any(w in line.lower() for w in
           ("persistence", "autoregress", "long-run", "long run",
            "true shift", "target shift", "mean rate", "structural", "ar(1)")):
        if budget219 <= 0:
            break
        s = line[:550]
        print(s)
        budget219 -= len(s)
print("\nCURRENT EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 219 — compact routing and dynamics recovery\n"
            "- Record estimates recovered: insulin approximately -2 and NE approximately +3 per rate unit; these are observational estimates, not revealed simulator truth.\n"
            "- Retrieving retained routing/replay objects after run218 output truncation.\n"
            "- No new weight edits; feed-forward compensation remains an untested hypothesis.\n")
```