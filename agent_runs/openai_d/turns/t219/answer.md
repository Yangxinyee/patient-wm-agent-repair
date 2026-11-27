The last gate summary accidentally used a scalar event label, so I’m discarding it. The persistence analysis still supports effects lasting several hours, but its long-lag estimates are uncertain. I’ll recompute the gates with the correct reading labels and inspect the available state caches. That should show whether lower-dose responses are being suppressed and whether broader factual calibration is possible.

```python
from pathlib import Path
import json, numpy as np
from scipy.special import ndtr

ev188 = np.asarray(r176_events)
assert ev188.shape == (r177_X.shape[1],)
print("CORRECT EVENT COUNTS", dict(zip(*np.unique(ev188, return_counts=True))))
W188 = np186(load("attr_heads.value_bin.0.weight"))
b188 = np186(load("attr_heads.value_bin.0.bias"))
for pair188 in ([107,109], [48,171]):
    g188 = r177_X @ ((W188[pair188[0]]+W188[pair188[1]])/2)
    g188 += (b188[pair188[0]]+b188[pair188[1]])/2
    d188 = r177_X @ ((W188[pair188[0]]-W188[pair188[1]])/2)
    d188 += (b188[pair188[0]]-b188[pair188[1]])/2
    print("\nPAIR", pair188)
    for e188 in np.unique(ev188):
        m188 = ev188 == e188
        print(e188,
              "gate quantiles", np.round(np.quantile(g188[:,m188],[.01,.1,.5,.9,.99]),2),
              "gate>0", round(float((g188[:,m188]>0).mean()),4),
              "contrast quantiles", np.round(np.quantile(d188[:,m188],[.01,.5,.99]),3))

print("\nCACHED FACTUAL DELTAS")
for label188, values188 in (
    ("stronger", c182_candidate_loss-c182_original_loss),
    ("checked fallback", c182_checked_loss-c182_original_loss)):
    print(label188, "overall", float(np.mean(values188)))
    for e188 in np.unique(ev188):
        print(e188, float(np.mean(values188[ev188 == e188])))
print("\nAVAILABLE FORWARD CACHES")
for folder188 in sorted(Path("/work").glob("fwd_*"), key=lambda p:int(p.name.split("_")[1])):
    p188 = folder188 / "summary.json"
    if p188.exists():
        s188 = json.loads(p188.read_text())
        print(folder188.name, s188.get("type"), s188.get("split"), s188.get("n_stays"),
              [p.name for p in folder188.glob("states*")])

print("\nFUNCTION REFERENCES")
for name188 in ("r145_hidden", "r145_head", "c181_fn", "local104"):
    f188 = globals()[name188]
    print(name188, "names", f188.__code__.co_names,
          "locals", f188.__code__.co_varnames,
          "small constants", [c for c in f188.__code__.co_consts
                              if isinstance(c,(str,int,float,tuple))][:20])

text188 = Path("/work/notes.md").read_text()
for heading188, limit188 in (
    ("## Runs 108",1800), ("## Runs 171",2700)):
    p188 = text188.find(heading188)
    print("\n", heading188, "\n", text188[p188:p188+limit188] if p188>=0 else "not found")
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 188: correct event-label audit\nRun 187 gate-by-event output used scalar r170_event and is invalid; recomputed with r176_events. No weights changed.\n")
```