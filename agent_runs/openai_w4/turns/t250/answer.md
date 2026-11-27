Earlier insulin transmission tests were nearly proportional, and the current norepinephrine decoder’s compression is modest. That may not explain the much larger rollout asymmetry. I’ll test the submitted channels with patient context held fixed. If doubled doses transmit much less per unit than half doses, the channel is responsible; if both transmit similarly, the distortion arises later in the rollout.

```python
import torch

r217_cpu_backup = dict(rW)
r217_transmission = []
try:
    rW = {n: torch.as_tensor(load(n), dtype=torch.float32) for n in NAMES}
    r217_base_E, r217_base_z = collect_selected187(indices102)
    r217_base_E = np.asarray(r217_base_E)
    r217_idx = np.asarray(indices102, dtype=int)

    for r217_drug, r217_flag, r217_decoder in [
        ("insulin", "has_insulin", joint_decoders193[0.0001]["ins"]),
        ("norepinephrine", "has_norepinephrine", joint_decoders193[0.0001]["ne"])
    ]:
        r217_mask = np.asarray(reference_bank125[r217_flag])[r217_idx].astype(bool)
        if r217_drug == "insulin":
            r217_qbase = np.asarray(gated_innovation94(raw87)).reshape(-1)
        else:
            r217_qbase = np.asarray(
                gated_ne185(raw87, calibrations192[(0.001, 100.0)]["coef"])
            ).reshape(-1)

        for r217_factor in [0.5, 2.0]:
            r217_raw = np.asarray(raw_cf87[(r217_drug, r217_factor)])
            r217_E, r217_z = collect_selected187(
                indices102, actions=r217_raw[:, 384:].astype(np.float32)
            )
            r217_delta = (
                (np.asarray(r217_E) - r217_base_E)
                @ np.asarray(r217_decoder)
            ).mean(axis=1)
            if r217_drug == "insulin":
                r217_qcf = np.asarray(gated_innovation94(r217_raw)).reshape(-1)
            else:
                r217_qcf = np.asarray(
                    gated_ne185(r217_raw, calibrations192[(0.001, 100.0)]["coef"])
                ).reshape(-1)
            r217_expected = r217_qcf - r217_qbase
            r217_x = r217_expected[r217_mask]
            r217_y = r217_delta[r217_mask]
            r217_transmission.append({
                "drug": r217_drug,
                "factor": r217_factor,
                "active_bundles": int(r217_mask.sum()),
                "raw_signal_change": float(r217_x.mean()),
                "transmitted_change": float(r217_y.mean()),
                "transmission_slope": float(
                    np.dot(r217_x, r217_y) / max(np.dot(r217_x, r217_x), 1e-12)
                ),
                "transmission_rmse": float(np.sqrt(np.mean((r217_y-r217_x)**2)))
            })
finally:
    rW = r217_cpu_backup

print(pd.DataFrame(r217_transmission).to_string(index=False))
print("\n=== CAUSAL-ESTIMATION OBJECTS FROM RUN 137 ===")
print(sorted(n for n in globals()
             if n.endswith("137") and not n.startswith("r217_")))
print("\nReplay source location:", replay_factual125.__code__.co_filename)
print("Head interface:", inspect.signature(head_features_fixed))
print("Head dependencies:", head_features_fixed.__code__.co_names)

with open("/work/notes.md", "a") as r217_file:
    r217_file.write(
        "\n## Run 217\n"
        "Tested the submitted insulin/NE channels on held-out-context half/double action encodings, "
        "holding patient context fixed. Results retained in r217_transmission. "
        "This measures channel transmission, not causal drug effects. "
        "CPU globals restored in finally; no saved weights changed.\n"
    )
```