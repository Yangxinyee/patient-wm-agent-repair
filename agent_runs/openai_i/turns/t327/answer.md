Gain 0.8 passed narrowly on the first 100 stays but failed on the next 100: factual NLL increased by 0.01046 nats. I cannot accept it. I’ll reduce the attention-routing changes while retaining the treatment signal. Success would mean lower factual loss without losing the improved dose responses.

```python
# Diagnose the failed candidate, then install a smaller routing edit.
paired284 = []
for k284 in [29, 31]:
    orig284 = pd.read_csv(f"/work/fwd_{k284}/factual_original.csv")
    edit284 = pd.read_csv(f"/work/fwd_{k284}/factual_edited.csv")
    keys284 = ["stay_id", "hour", "event", "observed"]
    pair284 = orig284.merge(edit284, on=keys284, suffixes=("_original", "_edited"))
    assert len(pair284) == len(orig284) == len(edit284)
    pair284["dNLL"] = pair284.nll_edited - pair284.nll_original
    paired284.append(pair284)
    print("cohort", k284, "dNLL", float(pair284.dNLL.mean()),
          "by event", pair284.groupby("event").dNLL.mean().round(6).to_dict())
pool284 = pd.concat(paired284, ignore_index=True)
print("pooled 200-stay dNLL", float(pool284.dNLL.mean()))

for model284 in ["original", "edited"]:
    dose284 = pd.read_csv(f"/work/fwd_30/dose_edit_{model284}.csv")
    reached284 = (
        dose284.reached if dose284.reached.dtype == bool
        else dose284.reached.astype(str).str.lower().isin(["true", "1"])
    )
    treated284 = dose284.loc[reached284]
    print("dose", model284,
          "reached-hour mean", treated284.groupby("edit")["shift"].mean().round(4).to_dict(),
          "stay-balanced mean",
          treated284.groupby(["edit", "stay_id"])["shift"].mean()
          .groupby(level=0).mean().round(4).to_dict())

with np.load("/work/neutral209.npz", allow_pickle=False) as file284:
    neutral284 = {k: file284[k].copy() for k in file284.files if k in NAMES}
with np.load("/work/corrected269.npz", allow_pickle=False) as file284:
    corrected284 = {k: file284[k].copy() for k in file284.files if k in NAMES}
assert "z_in.weight" in corrected284 and "z_in.weight" in neutral284
assert any(k.startswith("core.prelude.0") for k in corrected284)
np.savez("/work/rejected_gain08_284.npz",
         **{k: as_numpy282(load(k)).copy() for k in edited()})

rho284, response_gain284 = 0.6, 1.0
candidate284 = {k: v.copy() for k, v in neutral284.items()}
for name284, value284 in corrected284.items():
    base284 = neutral284.get(name284)
    if base284 is None:
        base284 = as_numpy282(load_original(name284))
    if name284.startswith("core.prelude.0."):
        candidate284[name284] = (
            base284 + rho284 * (value284 - base284)
        ).astype(np.float32)
    elif name284 == "z_in.weight":
        candidate284[name284] = (
            base284 + response_gain284 * (value284 - base284)
        ).astype(np.float32)
    else:
        # Retain the empty-action gate correction.
        candidate284[name284] = value284.astype(np.float32)

for name284 in list(edited()):
    drop_edit(name284)
for name284, value284 in candidate284.items():
    assert tuple(value284.shape) == tuple(SHAPES[name284])
    save_edit(name284, torch.from_numpy(value284.copy()).float())
np.savez("/work/blended284.npz", **candidate284)
print("Installed provisional routing blend", rho284,
      "response gain", response_gain284, "tensors", sorted(candidate284))

with open("/work/notes.md", "a") as log284:
    log284.write(
        "\n## Python 284 — reject gain-0.8 on independent factual cohort\n"
        f"- First cohort dNLL +0.004965; second +0.010455; pooled {pool284.dNLL.mean():+.6f}.\n"
        "- Saved rejected_gain08_284; it is not an acceptable final candidate.\n"
        "- Installed blended284 provisionally: prelude deltas scaled to 0.6, response gain 1.0.\n"
        "- Retained the actual-empty-action gate correction. No gradient training.\n"
        "- Hypothesis: reduce routing overhead while preserving treatment transport; requires GPU validation.\n"
    )
```