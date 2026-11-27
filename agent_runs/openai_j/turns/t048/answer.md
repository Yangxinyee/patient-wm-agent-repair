The architecture confirms that dose inputs affect the shared state, so this edit could change more than glucose and MAP predictions. The output was truncated before the stay IDs appeared. I’ll retrieve those IDs and a compact causal-evidence summary next; paired rollouts will then show whether the factual-safe probe actually improves dose responses.

```python
print("PROBE_STAY_IDS =", probe_ids)

for filename in ["dynamic_iv.csv", "sensor_iv.csv", "proxy_iv.csv", "kernel_targets.csv"]:
    frame = pd.read_csv(Path("/work") / filename)
    print("\n", filename, "shape =", frame.shape)
    print(frame.head(6).to_string(index=False))

print("\n=== IDENTIFICATION NOTES, BOUNDED EXCERPT ===")
print(notes_path.read_text()[5000:9500])

with notes_path.open("a") as f:
    f.write(
        "\n- Run 41: bounded recovery after run 40 output truncation. "
        "Architecture confirms the action-input probe affects shared state; "
        "all-head effects remain a concern. No further edits.\n"
    )
```