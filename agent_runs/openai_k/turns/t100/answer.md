The correction targets match the record-derived one-hour effects, and no obvious sign error appeared. But the local factual NLL is much higher than the earlier forward-tool baseline, so its preservation result needs verification. I’ll audit the saved factual predictions and local indexing before submitting a candidate; agreement would make the local tests trustworthy.

```python
import os, dis, inspect, numpy as np, pandas as pd

print("FORWARD ARTIFACTS")
for d94 in sorted(os.listdir("/work")):
    if d94.startswith("fwd_"):
        print(d94, os.listdir("/work/" + d94))
        for file94 in os.listdir("/work/" + d94):
            if file94.startswith("factual_") and file94.endswith(".csv"):
                path94 = "/work/" + d94 + "/" + file94
                frame94 = pd.read_csv(path94)
                print(" columns:", frame94.columns.tolist())
                print(frame94.head(4).to_string(index=False))
                print(" rows:", len(frame94))
                for col94 in frame94:
                    if "nll" in col94.lower():
                        print(" mean", col94, frame94[col94].mean())

print("\nCOMPACT LOCAL FUNCTION AUDIT")
for name94 in ("r59_core", "r59_features", "dataset80"):
    fn94 = globals().get(name94)
    if not inspect.isfunction(fn94):
        continue
    print("\n", name94, inspect.signature(fn94))
    print("references:", fn94.__code__.co_names)
    print("constants:", repr(fn94.__code__.co_consts)[:1000])
    instructions94 = list(dis.get_instructions(fn94))
    for ins94 in instructions94[:100]:
        print(ins94.offset, ins94.opname, ins94.argrepr)
    if len(instructions94) > 100:
        print("... total instructions:", len(instructions94))

print("\nINDEX/DATA OBJECTS")
for name94 in ("df", "train", "train_df", "panel", "states", "states59",
               "future_index59", "observed59", "rows80", "lookup80",
               "keys59", "index59", "bins59", "mask59"):
    if name94 in globals():
        x94 = globals()[name94]
        if isinstance(x94, pd.DataFrame):
            print(name94, x94.shape, x94.columns.tolist()[:20])
        elif isinstance(x94, (np.ndarray, torch.Tensor)):
            print(name94, tuple(x94.shape), np.asarray(x94).reshape(-1)[:10])
        elif isinstance(x94, dict):
            print(name94, "dict", len(x94), list(x94)[:5])
        else:
            print(name94, repr(x94)[:350])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 94: factual reconstruction audit\n"
            "- Drug response targets and signs agree with record identification.\n"
            "- Local original NLL (~5.06) differs from the earlier external baseline "
            "(~1.63); auditing saved predictions and index contracts before trusting "
            "local preservation scores.\n"
            "- No weights edited. Python runs: 94; forward passes: 4.\n")
```