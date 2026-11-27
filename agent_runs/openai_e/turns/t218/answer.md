The insulin baseline costs 0.00464 nats on the first tuning sample, leaving very little margin there. Several NE edits exceeded the limit, though one reduced trial stayed below it on another sample. I’ll inspect their full dose readouts and recover the prelude helpers. A promising NE correction should improve both dose directions without disrupting the insulin response.

```python
import json, inspect, re, numpy as np
from pathlib import Path

print("COMPLETE DOSE READOUT TABLE")
for n188 in [1, 5, 7, 10, 12, 15, 18]:
    r188 = json.loads(Path(f"/work/fwd_{n188}/summary.json").read_text())
    out188 = {}
    for m188, z188 in r188.get("results", {}).items():
        out188[m188] = {k: v["mean_shift"]
                       for k, v in z188.get("edits", {}).items()}
    print("fwd_"+str(n188), out188)

print("\nINDEPENDENT-CHANNEL CAUSAL CHECK")
ss188 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
for n188 in [63, 74, 75, 76, 154, 155, 156]:
    for s188 in ss188:
        if re.match(rf"## Run {n188}(?:\D|$)", s188):
            print(s188[:700].strip())

print("\nEDIT NORMS AND RANKS")
base188 = np.load("/work/baseline77.npz")["delta"]
print("baseline match error:",
      float(np.max(np.abs(load("core.b_in.weight")
                          - load_original("core.b_in.weight") - base188))))
for fn188 in ["ne_reduced107.npz", "ne_candidate151.npz",
              "ne_candidate154.npz", "ne_candidate155.npz"]:
    with np.load("/work/"+fn188, allow_pickle=False) as a188:
        print(fn188, [(k, a188[k].shape) for k in a188.files])
        if "ne_delta" in a188.files:
            sv188 = np.linalg.svd(a188["ne_delta"], compute_uv=False)
            print(" norm=", round(float(np.linalg.norm(a188["ne_delta"])), 3),
                  "leading singular values=", np.round(sv188[:4], 3))

print("\nPAIRED TRAINING STATE SCHEMA")
for m188 in ["original", "edited"]:
    with np.load(f"/work/fwd_22/states_{m188}.npz") as a188:
        print(m188, [(k, a188[k].shape) for k in a188.files])

print("\nRECONSTRUCTION FUNCTION NAMES")
fns188 = []
for k188, v188 in list(globals().items()):
    if inspect.isfunction(v188) and any(
        s in k188.lower() for s in
        ["prelude", "bundle", "recon", "context", "core", "encode", "inputs"]
    ):
        fns188.append((k188, v188.__code__.co_varnames[:8]))
print(fns188)

with open("/work/notes.md", "a") as f188:
    f188.write("\n## Run 188: recover full dose effects and reconstruction entry points\n"
               "Compared edited rollout readouts rather than truncated summaries. "
               "Insulin baseline remains installed. Preparing a repaired-state NE "
               "anchoring test; no weights changed. Python runs: 188; forwards: 22.\n")
```