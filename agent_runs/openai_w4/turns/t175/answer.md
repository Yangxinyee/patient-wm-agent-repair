This is still the run-154 display failure; the replay baseline has not been produced. I will run the baseline before any diagnostics, then inspect indexing without assuming its structure. If dose changes reach the input channel correctly but their response fades, that will support the persistence hypothesis; otherwise I will revisit the channel.

```python
import time, inspect, dis, numpy as np, torch

weights_current155 = {name: torch.as_tensor(load(name), dtype=torch.float32)
                      for name in NAMES}
t155 = time.time()
baseline155 = replay_factual125(weights_current155)
print("=== CURRENT-EDIT REPLAY BASELINE ===")
print("seconds:", round(time.time()-t155, 2))
print(brief154(baseline155))

print("\n=== INPUT-CHANNEL DOSE SLOPES ===")
for drug, obj in channel_probe147.items():
    rate = np.asarray(obj["rate"])
    q0 = np.asarray(obj["results"][1.0]["q"])
    for multiplier in [0.5, 2.0, 3.0]:
        qm = np.asarray(obj["results"][multiplier]["q"])
        dr = (multiplier-1.0)*rate
        good = np.isfinite(q0) & np.isfinite(qm) & np.isfinite(dr) & (np.abs(dr)>1e-6)
        dq = qm[good]-q0[good]
        print(drug, multiplier, "n", good.sum(),
              "mean_dq", round(float(dq.mean()), 5),
              "slope", round(float(np.dot(dq, dr[good]) / np.dot(dr[good], dr[good])), 5))
    idx = np.asarray(obj["indices"], dtype=int)
    print("indices/hours/insulin/NE:",
          list(zip(idx[:4], repair_bank["hour"][idx[:4]],
                   repair_bank["has_insulin"][idx[:4]],
                   repair_bank["has_norepinephrine"][idx[:4]])))

def compact_dis155(fn, stop_name=None):
    pieces = []
    for ins in dis.get_instructions(fn):
        if stop_name and ins.opname == "LOAD_GLOBAL" and ins.argval == stop_name:
            break
        if ins.opname not in ["RESUME", "CACHE", "PRECALL", "PUSH_NULL", "NOP"]:
            pieces.append(f"{ins.offset}:{ins.opname}({ins.argrepr})")
    return " ".join(pieces)

print("\n=== PRE-CONTEXT EXACT OPERATIONS ===")
print(compact_dis155(pre_context_repair)[:2400])
print("\n=== TRANSITION INPUT/PRELUDE OPERATIONS ===")
print(compact_dis155(transition73, stop_name="memory67")[:4000])
print("\n=== HEAD INTERFACE ===")
print("head_features_fixed dependencies:", head_features_fixed.__code__.co_names)
print("head_features_fixed constants:", brief154(head_features_fixed.__code__.co_consts))
print("probe_means95 operations:", compact_dis155(probe_means95)[:1200])
print("\n=== READING-EVALUATION FUNCTIONS ===")
for name, fn in list(globals().items()):
    if inspect.isfunction(fn) and any(s in name.lower() for s in ["nll", "reading", "means"]):
        print(name, inspect.signature(fn))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 155\nRun 154 recovered estimated equal-stay doubling targets "
            "(insulin glucose -13.2008; NE MAP +3.6087) but failed by treating Python's "
            "builtin next as an index array. Computed replay baseline before diagnostics "
            "and inspected exact indexing rather than guessing it. No weights changed.\n")
```