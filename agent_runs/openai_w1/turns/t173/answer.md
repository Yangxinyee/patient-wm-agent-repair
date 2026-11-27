The estimated effects persist for several hours, so one-step slopes understate the likely rollout changes. The saved edit was an untested fivefold amplification of the numeric dose and rate channels. I’ll test it as a reversible probe: stronger, correctly signed responses would support this pathway, while excessive factual loss would rule out using it unchanged.

```python
numeric_probe_scale = 5.0
numeric_probe_weight = torch.as_tensor(
    load_original("embed.numeric.weight")).clone()
numeric_probe_weight[:, 2] *= numeric_probe_scale
numeric_probe_weight[:, 4] *= numeric_probe_scale
save_edit("embed.numeric.weight", numeric_probe_weight)

probe_tuning_ids = sorted(
    pd.read_csv("/work/fwd_5/dose_edit_original.csv")["stay_id"].unique().tolist()
)
print("Current edits:", edited())
print("=== NEXT FORWARD REQUEST ===")
print(json.dumps({
    "type": "dose_edit",
    "model": "both",
    "split": "tuning",
    "stays": [int(x) for x in probe_tuning_ids]
}))

print("\n=== RECORD DOCUMENTATION ===")
print(open("/data/README.md").read())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 159\n"
            "Preserved run-154 edit was exactly 5x original numeric columns "
            "2 and 4, not the earlier 1.5x centered probe. Installed this "
            "simple 5x scaling as a reversible GPU experiment. Baseline "
            "enriched tuning shifts: insulin x2 -1.7733, half +0.8161; "
            "NE x2 +0.0479, half -0.4813. Estimated six-hour impulse "
            "responses imply larger accumulated effects. "
            "Counts: 159 Python runs, 8 forward passes.\n")
```