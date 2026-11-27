Glucose-finger accounts for most of the NLL deterioration, even though its predicted mean is closer to the observed mean. That argues against a simple mean-bias fix. I’ll inspect the gate and logit tilt next; if they behave regularly, a shorter persistence trial is the safer change.

```python
print("=== GATE / TILT DIAGNOSTICS FROM NOTEBOOK ===")
for heading in ["Runs 153", "Run 155", "Run 158", "Run 160", "Run 170", "Run 184"]:
    positions = [i for i, s in enumerate(nl) if heading in s]
    for pos in positions:
        end = next((j for j in range(pos+1, len(nl))
                    if nl[j].startswith("## ")), min(pos+12, len(nl)))
        print("\n".join(nl[pos:end])[:1600])

print("=== VALUE LOGIT DIRECTIONS ===")
for sensor, (control, partner) in sensor_neurons.items():
    print(sensor, np.round(ow280[:, control].numpy(), 3).tolist())

print("=== BIN TABLE ===")
print("codes:", VOCAB["codes"])
centres281 = VOCAB.get("value_bin_centres")
print(str(centres281)[:1900])

print("=== TRAINING HEAD INPUTS ===")
print("rows columns:", list(rows274.columns))
print(rows274.head(2).to_string(index=False))
pre281 = np.asarray(X275) @ hw280.numpy().T + hb280.numpy()
for sensor, (control, partner) in sensor_neurons.items():
    if "event" in rows274:
        mask281 = rows274["event"].eq(sensor).to_numpy()
    else:
        mask281 = np.ones(len(rows274), dtype=bool)
    print(sensor, "n", int(mask281.sum()),
          "control", np.round(np.quantile(pre281[mask281, control],
                                         [.01, .1, .5, .9, .99]), 4).tolist(),
          "partner", np.round(np.quantile(pre281[mask281, partner],
                                         [.01, .1, .5, .9, .99]), 4).tolist(),
          "signal", np.round(np.quantile(
              pre281[mask281, control] - pre281[mask281, partner],
              [.01, .1, .5, .9, .99]), 4).tolist())
print("test_ids264:", test_ids264[:5], len(test_ids264))
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart recovery, runs 277–281\n"
            "- Recovered architecture, provisional record-derived targets, and private-channel layout.\n"
            "- Current tau=1500 recentered candidate still fails tuning-A NLL by +0.00617.\n"
            "- Glucose-finger accounts for most deterioration despite improved aggregate mean; "
            "an intercept-only repair is not supported by this diagnostic.\n"
            "- No weights changed during recovery; candidate remains unaccepted.\n")
```
