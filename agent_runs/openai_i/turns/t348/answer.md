The last run failed because two intermediate arrays had incompatible sizes; it did not validate the edit. I’ll recover the current checkpoint and record-derived targets, then use GPU validation rather than spend the remaining runs repairing that diagnostic. The candidate must preserve the dose responses while keeping factual NLL within 0.005 nats of the original.

```python
import os, re, json, glob
import numpy as np
import pandas as pd
import torch

def as_numpy297(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== DATA README ===")
print(open("/data/README.md").read())

notes297 = open("/work/notes.md").read()
lines297 = notes297.splitlines()
pattern297 = re.compile(
    r"causal.{0,35}(target|effect|estimate)|"
    r"(target|effect|estimate).{0,35}causal|"
    r"record.based|instrument|randomiz|"
    r"true effects|ground.truth|"
    r"insulin.{0,60}(target|estimate)|"
    r"norepinephrine.{0,60}(target|estimate)",
    re.I
)
hits297 = [i for i, s in enumerate(lines297) if pattern297.search(s)]
print("=== RECORD-DERIVED EVIDENCE ===")
selected297 = set()
for i in hits297[:20]:
    selected297.update(range(max(0, i-1), min(len(lines297), i+3)))
evidence297 = "\n".join(lines297[i] for i in sorted(selected297))
print(evidence297[:7000])

print("=== CURRENT EDIT / CHECKPOINT AGREEMENT ===")
checkpoint297 = np.load("/work/protected_head295.npz", allow_pickle=False)
print("Checkpoint keys:", [(k, checkpoint297[k].shape) for k in checkpoint297.files])
for name297 in edited():
    arr297 = as_numpy297(load(name297))
    orig297 = as_numpy297(load_original(name297))
    match297 = (
        float(np.max(np.abs(arr297-checkpoint297[name297])))
        if name297 in checkpoint297.files else None
    )
    print(name297, "delta_RMS", float(np.sqrt(np.mean((arr297-orig297)**2))),
          "checkpoint_max_difference", match297)

print("=== RECENT FORWARD FILES ===")
for idx297 in range(33, 40):
    directory297 = f"/work/fwd_{idx297}"
    if os.path.isdir(directory297):
        print(idx297, sorted(os.listdir(directory297)))
        for model297 in ("original", "edited"):
            path297 = f"{directory297}/factual_{model297}.csv"
            if os.path.exists(path297):
                frame297 = pd.read_csv(path297)
                nllcols297 = [
                    c for c in frame297.columns
                    if "nll" in c.lower()
                    and pd.api.types.is_numeric_dtype(frame297[c])
                ]
                print(model297, "rows", len(frame297),
                      "NLL means", frame297[nllcols297].mean().to_dict())
                if idx297 == 34:
                    print("Columns:", list(frame297.columns))

sample297 = pd.read_csv("/data/tuning.csv", nrows=3)
staycol297 = next(c for c in sample297.columns if "stay" in c.lower())
tuning_ids297 = pd.read_csv(
    "/data/tuning.csv", usecols=[staycol297]
)[staycol297].drop_duplicates().tolist()
tuning_cohorts297 = [tuning_ids297[i:i+100] for i in range(0, len(tuning_ids297), 100)]
print("=== TUNING COHORTS ===")
for i297, ids297 in enumerate(tuning_cohorts297):
    print("Cohort", i297, json.dumps(ids297))

print("=== REUSABLE KERNEL OBJECTS ===")
names297 = [
    k for k in list(globals())
    if not k.endswith("297")
    and re.search(r"29[3456]|269|targets|causal|cohort", k, re.I)
]
for k297 in names297:
    obj297 = globals()[k297]
    shape297 = getattr(obj297, "shape", None)
    description297 = (
        str(shape297) if shape297 is not None
        else ("callable" if callable(obj297) else type(obj297).__name__)
    )
    print(k297, description297)

with open("/work/notes.md", "a") as log297:
    log297.write(
        "\n## Python 297 — final validation recovery\n"
        "- Run296 failed on a 7-versus-4 intermediate-array mismatch; its diagnostic is inconclusive.\n"
        "- Recovered architecture, record-derived evidence, checkpoint agreement, and tuning cohorts.\n"
        "- No weights changed. Protected-head candidate still requires independent GPU factual and rollout validation.\n"
    )
```