The retained baseline predicts insulin shifts of −12.71 and +4.34, versus record-derived targets around −13.6 and +6.8. Its factual NLL increase is +0.00278, leaving little room for leakage. I’ll inspect the corrected features from runs 94–95 and the NE estimates; low factual leakage with balanced dose sensitivity would justify another reversible trial.

```python
lines99 = []

lines99.append("CORRECTED FEATURE OBJECTS")
for n99, o99 in list(globals().items()):
    if n99.endswith(("93", "94", "95", "96")) and not n99.startswith("_"):
        if hasattr(o99, "shape"):
            lines99.append(f"{n99}: {type(o99).__name__} {o99.shape}")
        elif isinstance(o99, dict):
            lines99.append(f"{n99}: dict keys {list(o99)[:10]}")
        elif isinstance(o99, (int, float, str, np.number)):
            lines99.append(f"{n99}: {str(o99)[:100]}")

lines99.append("\nLOCAL HELPER CONTRACTS")
for n99 in ("prelude_hidden93", "posterior94", "prelude46", "edited_action46",
            "core_update_local", "coda46", "next_means45", "recon_ctx"):
    o99 = globals()[n99]
    c99 = o99.__code__
    lines99.append(f"{n99}: args={c99.co_varnames[:c99.co_argcount]}, "
                   f"names={c99.co_names}, "
                   f"constants={[x99 for x99 in c99.co_consts if isinstance(x99, (str, int, float))]}")

lines99.append("\nSAVED EDIT CONTENTS")
for p99 in ("/work/baseline77.npz", "/work/insulin_candidate63.npz",
            "/work/ne_candidate82.npz", "/work/ne_detector81.npz"):
    with np.load(p99) as a99:
        lines99.append(f"{p99}: { {k99: a99[k99].shape for k99 in a99.files} }")

lines99.append("\nNE IDENTIFICATION NOTES")
for s99 in sections97[1:]:
    title99 = s99.splitlines()[0]
    if any(f"Run {n99}:" in title99 for n99 in (69, 70, 71, 72, 73, 74, 75, 76)):
        lines99.append(s99[:800])

print("\n".join(lines99)[:8800])

# Match the next actually recorded arterial MAP, rather than drug hour + 1.
pred_paths99 = sorted(Path("/work").glob("fwd_*/factual_original.csv"))
pred99 = pd.concat([pd.read_csv(p99) for p99 in pred_paths99], ignore_index=True)
pred99 = pred99.drop_duplicates(["stay_id", "hour", "event"])
maps99 = pred99[pred99.event.eq("map_arterial")].copy()
drugs99 = tuning97[tuning97.event.eq("norepinephrine")].copy()
drugs99 = drugs99[drugs99.stay_id.isin(maps99.stay_id)]
matched99 = pd.merge_asof(
    drugs99.sort_values("hour"),
    maps99[["stay_id", "hour", "observed", "predicted_mean"]]
        .rename(columns={"hour": "map_hour"}).sort_values("map_hour"),
    left_on="hour", right_on="map_hour", by="stay_id",
    direction="forward", tolerance=2.0
)
ok99 = matched99.predicted_mean.notna()
print("\nCORRECTED TIME MATCH:", len(matched99), "NE records;",
      int(ok99.sum()), "next arterial readings within two hours")
print(matched99.loc[ok99, ["stay_id", "hour", "map_hour", "rate",
                         "observed", "predicted_mean"]].head(4).to_string(index=False))

with open("/work/notes.md", "a") as f99:
    f99.write("\n## Run 99: corrected timing and compact feature recovery\n"
              f"Next recorded arterial matching found {int(ok99.sum())}/{len(matched99)} NE records "
              "within two hours, avoiding fractional-hour equality. Recovering corrected features "
              "and record-derived NE dynamics. No weights changed. Python runs: 99; forwards: 12.\n")
```