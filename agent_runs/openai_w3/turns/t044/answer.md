Full bundles still disagree, and genuinely single-atom non-drug bundles do too, so omitted companion readings are not the main explanation. Drug starts remain exact. I’ll now check the architecture’s requirement that an empty action bundle produces zero; a nonzero cached action vector could explain this split.

```python
import numpy as np, torch

e39, u39, err39_old = r38_results["clear_numeric"]
idx39 = r38_idx
drug39 = r38_drug
print("Action-vector norms, no-drug bundles:",
      torch.linalg.vector_norm(u39[idx39[~drug39]], dim=1).min().item(),
      torch.linalg.vector_norm(u39[idx39[~drug39]], dim=1).max().item())
print("Bundle metadata examples:", repr(returned38[2][:5])[:800])

u39_zeroed = u39.clone()
u39_zeroed[np.flatnonzero(
    ~(np.asarray(S["has_insulin"]) | np.asarray(S["has_norepinephrine"]))
)] = 0
z39 = posterior_mean(r38_ctx, e39[idx39], u39_zeroed[idx39])
err39 = ((z39 - r38_target) ** 2).mean(dim=1).numpy()

for label39, mask39 in (
    ("all", np.ones(len(idx39), dtype=bool)),
    ("single non-drug", r38_single & ~drug39),
    ("multiple non-drug", ~r38_single & ~drug39),
    ("single drug", r38_single & drug39),
    ("multiple drug", ~r38_single & drug39),
):
    print(label39, "n", int(mask39.sum()),
          "before", float(err39_old[mask39].mean()),
          "zero-empty-action", float(err39[mask39].mean()))

print("\nFirst single-measurement comparisons:")
shown39 = 0
for pos39 in np.flatnonzero(r38_single & ~drug39):
    i39 = idx39[pos39]
    rows39 = state_records[
        (state_records.stay_id == int(S["stay_id"][i39])) &
        (np.abs(state_records.hour - float(S["hour"][i39])) < 0.0002)
    ]
    if len(rows39) == 1 and rows39.iloc[0]["event"] in (
        "glucose_finger", "glucose_lab", "map_arterial", "map_cuff"
    ):
        print(i39, rows39.iloc[0]["event"], rows39.iloc[0]["value"],
              "before", float(err39_old[pos39]),
              "zero-action", float(err39[pos39]))
        shown39 += 1
        if shown39 == 8:
            break

print("\nEmpty-vector live names:")
for name39, obj39 in list(globals().items()):
    if ("empty" in name39.lower() or name39 in ("u0", "e0")) and isinstance(obj39, torch.Tensor):
        print(name39, tuple(obj39.shape), "norm", obj39.norm().item())

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 39; forwards 3. Correct bundle cardinality did not resolve observation errors. Testing required zero action vector for empty action bundles, while drug-start matches remain exact. No edits.\n")
```