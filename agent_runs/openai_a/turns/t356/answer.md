The last batch again showed stronger insulin responses, with little change to norepinephrine predictions. I will retain the validated weights and now pool all 400 rollouts, check factual fit and tensor integrity, and recover the causal-target notes for the final summary. Inconsistent coverage, weights, or estimates would invalidate that summary; no further forward tests remain.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

rollout_frames292 = []
factual_frames292 = []
sources292 = {"dose_edit": [], "factual": []}

for k in range(41, 51):
    folder = Path(f"/work/fwd_{k}")
    if (folder / "dose_edit_edited.csv").exists():
        sources292["dose_edit"].append(k)
        for model in ("original", "edited"):
            df = pd.read_csv(folder / f"dose_edit_{model}.csv")
            df["model"] = model
            df["forward"] = k
            rollout_frames292.append(df)
    if k <= 46 and (folder / "factual_edited.csv").exists():
        sources292["factual"].append(k)
        o = pd.read_csv(folder / "factual_original.csv")
        e = pd.read_csv(folder / "factual_edited.csv")
        assert len(o) == len(e)
        for c in ("stay_id", "hour", "event", "observed"):
            assert o[c].equals(e[c]), (k, c)
        q = o[["stay_id", "hour", "event"]].copy()
        q["original_nll"] = o.nll
        q["edited_nll"] = e.nll
        q["delta"] = e.nll - o.nll
        factual_frames292.append(q)

print("=== VALIDATION SOURCES ===")
print(sources292)
all_rollouts292 = pd.concat(rollout_frames292, ignore_index=True)
assert all_rollouts292.stay_id.nunique() == 400
assert not all_rollouts292.duplicated(["model", "stay_id", "hour", "edit"]).any()
if all_rollouts292.reached.dtype == bool:
    reached292 = all_rollouts292.reached
else:
    reached292 = all_rollouts292.reached.astype(str).str.lower().isin(["true", "1"])
readouts292 = all_rollouts292.loc[reached292].copy()
assert readouts292.hour.between(13, 24).all()

summary292 = readouts292.groupby(["model", "edit"]).agg(
    mean_shift=("shift", "mean"),
    n_stay_hours=("shift", "size"),
    n_stays=("stay_id", "nunique"))
print("=== EXACT POOLED 400-STAY READOUTS (REACHED STAY-HOURS) ===")
print(summary292.to_string())

rng292 = np.random.default_rng(292)
intervals292 = {}
for (model, edit), group in readouts292.groupby(["model", "edit"]):
    clusters = group.groupby("stay_id")["shift"].agg(["sum", "size"])
    ix = rng292.integers(0, len(clusters), (4000, len(clusters)))
    bs = (clusters["sum"].to_numpy()[ix].sum(axis=1) /
          clusters["size"].to_numpy()[ix].sum(axis=1))
    intervals292[(model, edit)] = np.quantile(bs, [.025, .975]).tolist()
print("Edited readout stay-bootstrap intervals:",
      {edit: ci for (model, edit), ci in intervals292.items() if model == "edited"})

full_factual292 = pd.concat(factual_frames292, ignore_index=True)
assert full_factual292.stay_id.nunique() == 400
assert len(full_factual292) == 18654
print("=== FULL400 FACTUAL FIT ===")
print("Original:", full_factual292.original_nll.mean())
print("Edited:", full_factual292.edited_nll.mean())
print("Paired increase:", full_factual292.delta.mean())
print(full_factual292.groupby("event").delta.agg(["size", "mean"]))

expected292 = {
    "attr_heads.value_bin.0.bias", "attr_heads.value_bin.0.weight",
    "attr_heads.value_bin.2.bias", "attr_heads.value_bin.2.weight"}
assert set(edited()) == expected292
assert all(np.array_equal(numpy291(load(n)),
                          numpy291(amplitude093_checkpoint[n]))
           for n in expected292)
print("Integrity: exactly four validated amplitude093 tensors; no weight changes.")

print("=== SUSTAINED EFFECT IDENTIFICATION NOTE ===")
for section in sections291:
    if section.startswith("## Sustained insulin target"):
        print(section[:2400])

print("=== TARGET HELPER RECOVERY ===")
f292 = globals().get("matched_causal_targets163")
if callable(f292):
    print("Signature:", inspect.signature(f292))
    print("Referenced names:", f292.__code__.co_names)
    print("Local names:", f292.__code__.co_varnames)
    for name in f292.__code__.co_names:
        obj = globals().get(name)
        if obj is not None and name not in ("np", "pd"):
            print("Reference:", name, type(obj).__name__, getattr(obj, "shape", None))

target_names292 = [
    n for n in list(globals())
    if ("target" in n.lower() or "causal" in n.lower())
    and not n.endswith(("291", "292"))]
print("Target-related variables:",
      [(n, type(globals()[n]).__name__, getattr(globals()[n], "shape", None))
       for n in target_names292[:35]])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Complete rollout validation; Python 292, forwards 50/50\n"
        "- No weights changed; current edit exactly matches amplitude093_checkpoint.\n"
        f"- Rollout sources: {sources292['dose_edit']}; all400 tuning stays covered.\n"
        "- Exact reported readout pools reached stay-hours, not unweighted stay means.\n"
        + summary292.to_string() + "\n"
        f"- Full400 factual paired NLL increase: {full_factual292.delta.mean():.8f}.\n"
        "- Additional200 training-stay increase: +0.0024419958.\n"
        "- Final choice remains validated amplitude093; causal-target audit pending.\n")
```
