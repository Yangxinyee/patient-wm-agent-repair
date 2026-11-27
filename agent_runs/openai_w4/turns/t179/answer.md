The records support sustained effects: insulin’s estimated response remains negative for several hours, and norepinephrine’s is positive. These estimates depend on the stated identification assumptions, but they are much larger than the edited rollouts. I’ll now test a single dose change through the replay to see whether the model loses the response between bundles.

```python
import dis, inspect, numpy as np, pandas as pd
from pathlib import Path

print("=== GPU ROLLOUT READOUTS, EQUAL WEIGHT PER STAY ===")
for path158 in sorted(Path("/work").glob("fwd_*/dose_edit_*.csv")):
    if path158.parent.name not in ["fwd_1", "fwd_7", "fwd_8", "fwd_10", "fwd_11"]:
        continue
    df158 = pd.read_csv(path158)
    use158 = df158[df158["reached"].astype(bool)]
    means158 = use158.groupby(["edit", "stay_id"])["shift"].mean().groupby("edit").agg(["mean", "count"])
    print(str(path158), means158.round(4).to_dict("index"))

print("\n=== REPLAY FUNCTION OPERATIONS ===")
lines158 = []
for ins158 in dis.get_instructions(replay_factual125):
    lines158.append(ins158.opname + (":" + ins158.argrepr if ins158.argrepr else ""))
print(" ".join(lines158)[:5800])

print("\n=== ACTION HELPERS ===")
for k158, v158 in list(globals().items()):
    if inspect.isfunction(v158) and (k158.startswith("rp_") or any(s in k158 for s in ["atom", "embed_repair"])):
        print(k158, inspect.signature(v158))
for fn158 in [rp_action_encoder, singleton_action_repair]:
    print(fn158.__name__, "dependencies", fn158.__code__.co_names,
          "constants", [x for x in fn158.__code__.co_consts
                        if isinstance(x, (str, int, float, tuple))])

print("\n=== RETAINED DOSE-PROBE OBJECTS ===")
for k158, v158 in list(globals().items()):
    if k158.endswith(("147", "148", "153", "154", "155")):
        if isinstance(v158, dict):
            desc158 = "keys=" + str(list(v158.keys())[:12])
        else:
            desc158 = str(getattr(v158, "shape", type(v158).__name__))
        if not inspect.isfunction(v158):
            print(k158, desc158)
print("repair_records sample:",
      repair_records[repair_records.event.isin(["insulin", "norepinephrine"])].head(3).to_dict("records"))

Path("/work/notes.md").open("a").write(
    "\n## Run 158\nThe 12-lag record kernels support sustained insulin lowering and NE raising effects "
    "under innovation-exogeneity/additive-kernel assumptions. Recovering exact replay/action construction "
    "for a one-dose persistence test. No weights changed.\n")
```
