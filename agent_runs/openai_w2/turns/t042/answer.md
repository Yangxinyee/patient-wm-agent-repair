The earlier estimates imply larger accumulated effects than the original rollout predicts, but they rely on uncertain persistence assumptions. I’ll inspect the hourly response and the relevant head tensors, then fit a small measurement-error state model to separate within-stay persistence from stable differences between stays. That should give us better-supported rollout targets.

```python
print("=== PANEL STRUCTURE ===")
print("panel columns:", panel.columns.tolist())
print(panel.head(12).to_string())
print("F columns:", F.columns.tolist())
print("head_data columns:", head_data.columns.tolist())
print(head_data.head(3).to_string())
print("repair config:", globals().get("repair_head_cfg"))
for key in ["S_reference", "S_probe", "repair_original_features",
            "repair_probe_features", "repair_delta_hidden", "probe_draw_store"]:
    obj = globals().get(key)
    if isinstance(obj, dict):
        print(key, "dict keys:", list(obj)[:12])
    else:
        print(key, type(obj).__name__, getattr(obj, "shape", None))

print("\n=== NUMERIC SPECIFICATIONS ===")
for item in VOCAB["numeric"]:
    print(item)

print("\n=== ORIGINAL HOURLY DOSE RESPONSE ===")
dose_baseline = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
reached_mask = dose_baseline["reached"].isin([True, 1, "True", "true", "1"])
print(dose_baseline[reached_mask]
      .groupby(["edit", "hour"]).agg(
          shift=("shift", "mean"), factual=("factual", "mean"),
          n=("stay_id", "size")).round(3).to_string())

print("\n=== TARGET TENSOR SHAPES ===")
for name in NAMES:
    if any(s in name for s in [
        "embed.numeric", "z_post", "c_out", "out_norm",
        "last_numeric", "attr_heads.value_bin", "embed_transition.value_bin"
    ]):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 34–37: recovered context\n"
            "Architecture reread. Clean insulin lab RD about -1.55/unit, "
            "rich-history estimate -1.97; NE estimates +3.5 to +3.9/unit. "
            "Original 100-stay rollout -3.226/+1.825/+1.373/-0.846. "
            "No edits. Persistence must separate stable stay heterogeneity "
            "from dynamic propagation; plan a small latent-state fit.\n")
```
