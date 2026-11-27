The insulin pair has one response neuron and one gate-only neuron, so a stricter gate can suppress halving and doubling differently. I’ll compare earlier rollouts and recover the factual scorer before screening lower gates or a symmetric response pair. The useful test is improved response in both directions while NLL stays within the limit.

```python
from pathlib import Path
from io import StringIO
import dis, numpy as np, pandas as pd

print("HISTORICAL REACHED-ROW DOSE SHIFTS")
hist263 = []
for p in sorted(Path("/work").glob("fwd_*/dose_edit_edited.csv"),
                key=lambda p: int(p.parent.name.split("_")[1])):
    d = pd.read_csv(p)
    if {"edit","shift","reached"}.issubset(d.columns):
        r = d[d["reached"].astype(str).str.lower().isin(["true","1"])]
        means = r.groupby("edit")["shift"].mean()
        hist263.append({"folder":p.parent.name,
                        **{k:round(float(means.get(k,np.nan)),3)
                           for k in ("insulin_x2","insulin_half","ne_x2","ne_half")}})
print(pd.DataFrame(hist263).to_string(index=False))

print("\nCONSTRUCTION NOTES 192 ONWARD")
nt263 = Path("/work/notes.md").read_text()
ss263 = nt263.find("## Run 192")
print(nt263[ss263:ss263+3400] if ss263 >= 0 else "section not found")

print("\nFACTUAL EVALUATION INTERFACES")
for k in sorted(list(globals())):
    v = globals()[k]
    if callable(v) and any(t in k.lower() for t in ("eval", "score", "factual")):
        try:
            print(k, str(inspect.signature(v))[:180], v.__code__.co_names[:22])
        except Exception:
            pass
for k in ("replay_factual125", "candidate258"):
    o = StringIO()
    dis.dis(globals()[k], file=o, depth=0)
    print(k, "TAIL\n", o.getvalue()[-1300:])
print("insulin pair biases", [(i,float(a262(load("z_post.0.bias"))[i]))
                               for i in (96,184)])
with open("/work/notes.md","a") as f:
    f.write("\n## Run 263\nHistorical GPU dose-shift comparison and recovery "
            "of factual scoring interfaces. One-sided insulin pair and gate "
            "suppression motivate a controlled gate screen. No weights saved.\n")
```
