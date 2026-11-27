The first batch increased NLL by 0.00505 nats, while the second increased it by 0.00282; pooled, these are below the limit. This is encouraging but incomplete. I’ll recover the remaining validation IDs and the record-based effect estimates. The candidate is worth retaining only if the full pooled fit passes and its rollout effects remain credible.

```python
final_paired_parts = []
for k in (37, 38):
    a, b = final_frames[k]["original"], final_frames[k]["edited"]
    keys = ["stay_id", "hour", "event", "observed"]
    assert a[keys].equals(b[keys])
    q = a[keys].copy()
    q["delta_nll"] = b["nll"].to_numpy() - a["nll"].to_numpy()
    final_paired_parts.append(q)
final_paired = pd.concat(final_paired_parts, ignore_index=True)
print("First 200 stays:", final_paired.stay_id.nunique(),
      "readings:", len(final_paired),
      "pooled delta:", final_paired.delta_nll.mean())
print("By event:", final_paired.groupby("event").delta_nll.agg(["count", "mean"]).to_dict())

for label, k in (("ROLLOUT_BATCH_1", 37), ("FACTUAL_BATCH_3", 35),
                 ("FACTUAL_BATCH_4", 36)):
    d = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    print(label, sorted(d.stay_id.unique().tolist()))

notes_recovery = open("/work/notes.md").read()
print("EARLY RECORD IDENTIFICATION NOTES:\n", notes_recovery[:6500])
for name in ("effects63", "targets158", "r244_targets", "target_lines"):
    if name in globals():
        v = globals()[name]
        print("RETAINED", name, "type", type(v).__name__, str(v)[:1500])

m = np.load("/work/margin_candidate.npz")
print("Current matches margin archive:",
      np.array_equal(np.asarray(load("core.b_in.weight")), m["core_b_in"]))
print("Correction norms:",
      {k: float(np.linalg.norm(m[k])) for k in
       ("glucose_correction", "map_correction")})

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 290: exact partial validation and record-evidence recovery\n"
            f"- Margin candidate forwards 37/38 pooled delta: {final_paired.delta_nll.mean():.9f} "
            f"over {len(final_paired)} readings from 200 stays.\n"
            "- Retrieved remaining disjoint factual batches and prior record-identification evidence.\n"
            "- No weights changed. Python runs: 290; forward passes: 38.\n")
```