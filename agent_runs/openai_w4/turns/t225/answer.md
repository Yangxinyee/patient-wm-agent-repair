The training scale tests show little added damage from the channel at zero NE-response strength, but worsening fit as the response grows. I’ll halve that response, preserving the insulin correction, and test it on held-out stays. Passing the NLL limit would support a strength-related explanation; failure would point to a broader channel problem.

```python
import torch
b_original199 = np.asarray(load_original("core.b_in.weight"), dtype=np.float64)
b_current199 = np.asarray(load("core.b_in.weight"), dtype=np.float64)
b_scale_one199 = np.asarray(preserved_b194) + np.outer(
    np.asarray(ne_direction194), np.asarray(joint194["ne"])
)
print("B CONSTRUCTION MAX ERROR",
      float(np.max(np.abs(b_current199 - b_scale_one199))))
assert np.allclose(b_current199, b_scale_one199, atol=2e-4, rtol=2e-6)

print("PREVIOUS GPU ROLLOUTS: eligible-hour means, then equal-stay means")
dose_paths199 = sorted(
    Path("/work").glob("fwd_*/dose_edit_*.csv"),
    key=lambda p: (int(p.parent.name.split("_")[-1]), p.name)
)
for p199 in dose_paths199[-10:]:
    d199 = pd.read_csv(p199)
    if "reached" in d199:
        d199 = d199[d199["reached"].eq(True)]
    if {"stay_id", "edit", "shift"}.issubset(d199.columns):
        equal199 = d199.groupby(["stay_id", "edit"])["shift"].mean().groupby("edit").mean()
        row199 = d199.groupby("edit")["shift"].mean()
        print(str(p199), "equal_stay", equal199.round(4).to_dict(),
              "row", row199.round(4).to_dict())

b_half199 = (
    np.asarray(preserved_b194) +
    0.5 * np.outer(np.asarray(ne_direction194), np.asarray(joint194["ne"]))
).astype(np.float32)
save_edit("core.b_in.weight", torch.from_numpy(b_half199))
print("SAVED NE RESPONSE SCALE", 0.5)
print("TRAIN REPLAY DELTA AT THIS SCALE", results194[0.5]["delta_original"])
print("VALIDATION STAYS", len(validation_ids195))
with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 199\nVerified submitted B_in equals preserved insulin connection plus scale-1 NE rank-one response. Saved NE response scale .5, all channel and insulin weights unchanged. Training replay dNLL at .5 is +.0022796 versus original. Failed scale-1 dual weights remain backed up in backup_dual198. GPU factual and rollout checks pending; prior rollout CSVs summarized using eligible-hour, equal-stay aggregation.\n")
```