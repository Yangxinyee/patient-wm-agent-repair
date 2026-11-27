The pilot increased factual NLL by 0.00591, just beyond the allowed limit. That rejects the 15% version. I’ll reduce it to 8% and then check whether it improves dose responses at all; if the response remains negligible, this correction is not worth its factual cost.

```python
def freeze_ops(specifications):
    frozen = []
    for specification in specifications:
        item = dict(specification)
        if item["op"] == "rank1":
            vector = item["vector"]
            item["vector"] = (np.load(Path("/work") / vector).copy()
                              if isinstance(vector, str) else np.asarray(vector).copy())
        frozen.append(item)
    return frozen

# Preserve vector contents, rather than relying on filenames that later edits may reuse.
repair_candidates["numeric_gain4"] = freeze_ops(numeric4_saved_ops)
repair_candidates["slot_shear_015"] = freeze_ops(
    json.loads(Path("/work/ops.json").read_text()))
numeric4_saved_ops = repair_candidates["numeric_gain4"]

pilot_strength = 0.08
pilot_reduced = np.eye(pilot_basis.shape[1]) + pilot_strength * (
    pilot_basis.T @ pilot_design["S"] @ pilot_basis)
replay_ops(numeric4_saved_ops)
apply_lowrank_left_transform("core.c_out.weight", pilot_basis, pilot_reduced)
repair_candidates["slot_shear_008"] = freeze_ops(
    json.loads(Path("/work/ops.json").read_text()))

f14o = pd.read_csv("/work/fwd_14/factual_original.csv")
f14e = pd.read_csv("/work/fwd_14/factual_edited.csv")
print("Factual columns:", list(f14o.columns))
print("15% candidate NLL increase:", f14e["nll"].mean()-f14o["nll"].mean())
print("Current candidate: numeric gain 4 plus slot shear strength", pilot_strength)
print("Approximate one-application reading-shift SD:",
      pilot_strength*pilot_design["predicted_shift"].std(axis=0))

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 14 / Python run 25: reduce candidate D\n")
    f.write("- Slot shear 0.15 plus numeric gain 4: factual NLL 1.64052 versus 1.63461, "
            "+0.00591, exceeding the limit; rejected at that strength.\n")
    f.write("- Reduced shear to 0.08. Next inspect dose rollouts before further tuning.\n")
    f.write("- repair_candidates now stores copied vector arrays; replay_ops restores candidates. "
            "pilot_basis and pilot_design define the shear.\n")
    f.write("- Python runs: 25; forward passes: 14.\n")
```