The record-derived working estimate is about +3.8 for doubled norepinephrine, whereas the current rollout gives +0.9. That gap does not justify blind amplification. I’ll examine the correction’s loss–response frontier and construction; better held-out response at comparable factual loss would justify a new candidate.

```python
import inspect, dis

print("MAP FRONTIER")
print(map172_frontier.to_string(index=False))
print("\nRAW MAP FRONTIER: LOWEST TUNING LOSS")
print(map164_frontier.sort_values("tuning_dNLL").head(6).to_string(index=False))

print("\nHELPER SIGNATURES")
for name188 in ["head177_make", "head177_eval", "wm_mc145", "wm_mc_raw146",
                "wm_meta145", "wm_rawloss152", "map172_tau"]:
    fn188 = globals()[name188]
    print(name188, inspect.signature(fn188))

print("\nHEAD CONSTRUCTION")
lines188 = {}
line188 = None
for ins188 in dis.get_instructions(head177_make):
    if ins188.starts_line is not None:
        line188 = ins188.starts_line
    lines188.setdefault(line188, []).append(
        ins188.opname + (" " + ins188.argrepr if ins188.argrepr else "")
    )
for line188, tokens188 in lines188.items():
    print(line188, " | ".join(tokens188))

print("\nMETADATA")
print(wm_tune_meta145.columns.tolist())
print(wm_tune_meta145.head(3).to_string(index=False))
for label188, mc188 in [("train", wm_train145), ("tuning", wm_tune145)]:
    raw188 = mc188["raw"]
    print(label188, "raw type", type(raw188).__name__)
    if isinstance(raw188, dict):
        print("raw keys", list(raw188.keys()))
    elif hasattr(raw188, "shape"):
        print("raw shape", raw188.shape)
```
