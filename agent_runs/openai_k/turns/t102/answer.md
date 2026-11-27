The local readout predicts the next integer hour, and the audit uses filtered bundle indices. That makes an indexing mistake plausible. I’ll rebuild the comparison from the factual CSV’s observed values, rather than reuse the old bin table. Matching means and NLLs would confirm the reconstruction and identify which earlier scores were misaligned.

```python
import numpy as np, pandas as pd, torch

print("SENSOR ORDER:", r49_sensors)
print("FIRST REGISTER:", r49_register[rows80[0]])
for sensor96 in r49_sensors:
    spec96 = num_spec0[(sensor96, "value")]
    print(sensor96, "edges", spec96["fine_bin_edges"])

mapped96 = lookup80[rows80]
assert np.all(mapped96 >= 0)
audit_parts96 = []
for j96, sensor96 in enumerate(r49_sensors):
    t96 = pd.DataFrame({
        "stay_id": future_index59.get_level_values(0),
        "hour": future_index59.get_level_values(1),
        "event": sensor96,
        "local_mean": M0_80[mapped96, j96],
        "local_position": mapped96
    })
    t96 = t96.merge(factual95, on=["stay_id", "hour", "event"], how="inner")
    b96 = 1 + np.searchsorted(
        num_spec0[(sensor96, "value")]["fine_bin_edges"],
        t96.observed.to_numpy(), side="left")
    p96 = probs0_80[t96.local_position.to_numpy(), j96, b96]
    t96["local_nll"] = -np.log(np.maximum(p96, 1e-30))
    t96["bin_from_record"] = b96
    t96["mean_error"] = t96.local_mean - t96.predicted_mean
    t96["nll_error"] = t96.local_nll - t96.nll
    audit_parts96.append(t96)
audit96 = pd.concat(audit_parts96, ignore_index=True)
print("\nEXACT-KEY COMPARISON")
print("readings:", len(audit96))
print(audit96.groupby("event").agg(
    local_nll=("local_nll", "mean"),
    external_nll=("nll", "mean"),
    mean_bias=("mean_error", "mean"),
    mean_RMS=("mean_error", lambda x: np.sqrt(np.mean(x*x))),
    nll_error_RMS=("nll_error", lambda x: np.sqrt(np.mean(x*x)))
).round(5).to_string())
print(audit96.head(8)[["stay_id", "hour", "event", "observed",
                       "bin_from_record", "local_mean", "predicted_mean",
                       "local_nll", "nll"]].to_string(index=False))

print("\nFRESH ORIGINAL READOUT CHECK")
ix96 = rows80[:24]
out96, h96 = r59_core(ix96)
features96 = r59_features(out96, h96, ix96)
print("returned shapes:", [tuple(v.shape) for v in features96])
fresh_mean96 = np.asarray(features96[0])
print("fresh/saved mean max difference:",
      np.max(np.abs(fresh_mean96 - M0_80[lookup80[ix96]])))
print("fresh/reconstructed state max differences:",
      float(torch.max(torch.abs(out96 - torch.tensor(sx[ix96, :192])))),
      float(torch.max(torch.abs(h96.reshape(-1, 768) -
                                torch.tensor(sx[ix96, 192:960])))))

print("\nREADOUT TAIL")
for ins96 in list(__import__("dis").get_instructions(r49_readouts))[-115:]:
    if ins96.opname not in {"COPY", "SWAP", "EXTENDED_ARG", "NOP", "POP_TOP"}:
        print(ins96.opname, ins96.argrepr)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 96: independent reading-key audit\n"
            "- Rebuilt local factual comparison from external CSV observations, "
            "matched by stay/hour/sensor; also recomputed original states and readouts.\n"
            "- This avoids trusting previous bin and mask alignment.\n"
            "- No edits. Python runs: 96; forward passes: 4.\n")
```