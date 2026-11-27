The center is indeed carried by ordinary features, so it can respond to changed doses. I will run a reversible constant-center trial to isolate that pathway; this is a diagnostic, not an accepted repair. If halving gains a response comparable to half the doubling response, centering likely explains much of the asymmetry. If not, I will restore the baseline and investigate transport or feedback.

```python
from safetensors.numpy import save_file as r249_save_file

r249_snapshot = {
    name: np.asarray(load(name), dtype=np.float32).copy()
    for name in edited()
}
r249_save_file(r249_snapshot, "/work/checkpoint_run249.safetensors")

r249_w0 = np.asarray(load("attr_heads.value_bin.0.weight")).copy()
r249_b0 = np.asarray(load("attr_heads.value_bin.0.bias")).copy()
r249_sensor_controls = {
    "glucose_finger": 171, "glucose_lab": 7,
    "map_arterial": 124, "map_cuff": 10
}
r249_constants = {}
for r249_sensor, r249_control in r249_sensor_controls.items():
    r249_partner = _pair_for_control[r249_control]
    r249_delta = r249_w0[r249_partner] - r249_w0[r249_control]
    r249_keep = np.zeros(192, dtype=np.float32)
    r249_keep[r248_private] = r249_delta[r248_private]
    r249_const = float(np.mean(head_audit[r249_sensor]["exposure"]))
    r249_w0[r249_partner] = r249_w0[r249_control] + r249_keep
    r249_b0[r249_partner] = r249_b0[r249_control] - r249_const
    r249_constants[r249_sensor] = r249_const

save_edit("attr_heads.value_bin.0.weight", r249_w0)
save_edit("attr_heads.value_bin.0.bias", r249_b0)
print("Installed diagnostic constant centers:", r249_constants)

print("\n=== PRIOR ROLLOUTS: equal-weight mean across reached stays ===")
for r249_num in (18, 22):
    r249_path = Path(f"/work/fwd_{r249_num}/dose_edit_edited.csv")
    if r249_path.exists():
        r249_df = pd.read_csv(r249_path)
        r249_reached = r249_df[r249_df["reached"]]
        print("forward folder", r249_num)
        print(r249_reached.groupby(["edit", "stay_id"])["shift"].mean()
              .groupby("edit").agg(["mean", "count"]).round(4))
        print("hours 13, 18, 24:")
        print(r249_reached[r249_reached.hour.isin([13, 18, 24])]
              .groupby(["edit", "hour"])["shift"].mean().round(4))

with open("/work/notes.md", "a") as r249_f:
    r249_f.write(
        "\n## Run 249: reversible centering-isolation trial\n"
        "- Saved all pre-trial edits in /work/checkpoint_run249.safetensors.\n"
        "- Replaced ordinary-feature exposure centers with sensor-specific "
        "constant factual exposure means; retained private-dose coefficients "
        "and value-head output strengths unchanged.\n"
        "- This is a diagnostic, not an accepted repair. GPU paired rollouts "
        "must establish whether removing center leakage reduces asymmetry; "
        "factual NLL must be tested before retention.\n"
    )
```
