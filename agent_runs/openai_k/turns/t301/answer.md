The MAP trial improves norepinephrine responses only modestly (+1.11/−0.36), despite failing factual fit. I’ll remove it and modestly strengthen the glucose correction. I’ll also inspect the readout helpers for a MAP route that preserves full glucose distributions, not just their means. The stronger glucose edit must still pass the factual limit.

```python
import dis
import io
import pandas as pd

np.savez("/work/rejected_map_trial.npz", b_in=current265)
r266_I = 1.15
r266_weight = (base265 + r266_I * np.asarray(r255_CI)).astype(np.float32)
save_edit("core.b_in.weight", torch.from_numpy(r266_weight))
np.savez("/work/glucose_115_candidate.npz", b_in=r266_weight)

print("Submitted: glucose strength", r266_I, "; MAP addition disabled.")
print("Edited tensors:", edited())

print("\n=== LOCAL CASES ===")
print(compact265(r245_cases))

print("\n=== ROUTING OBJECTS ===")
for name, obj in list(globals().items()):
    if re.match(r"^r24[45]_", name):
        print(name, compact265(obj))

print("\n=== EXACT READOUT INTERFACES ===")
for name in ["r244_read_pre", "r179_capture_head"]:
    obj = globals()[name]
    buf = io.StringIO()
    dis.dis(obj, file=buf)
    print(name)
    print(buf.getvalue()[:3600])
print("r49_readouts args:",
      r49_readouts.__code__.co_varnames[:r49_readouts.__code__.co_argcount])
print("r49_readouts globals:", r49_readouts.__code__.co_names)

print("\n=== TUNING ID CHECK ===")
header266 = pd.read_csv("/data/tuning.csv", nrows=0)
print("Columns:", header266.columns.tolist())
staycol266 = next(c for c in header266.columns if c in ["stay", "stay_id", "icustay_id"])
ids266 = sorted(pd.read_csv("/data/tuning.csv", usecols=[staycol266])[staycol266].unique().tolist())
print("Number of stays:", len(ids266))
print("First / last:", ids266[:3], ids266[-3:])
print("Next factual validation batch:", ids266[100:200])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 266: reject inefficient MAP route; strengthen glucose modestly\n"
            "- Forward 23 rejected MAP trial: insulin -10.779/+4.926; NE +1.113/-0.359.\n"
            "- MAP improvement is too small for its +0.00909 total factual-loss cost.\n"
            "- Submitted glucose-only correction at strength 1.15; MAP disabled.\n"
            "- Recovering full-distribution readout interfaces for a better MAP route.\n"
            "- Python runs: 266; forward passes: 23.\n")
```