The recalibrated channel transmits double doses reasonably well and preserves the existing insulin signal closely. I’ll now test MAP-response strengths against factual replay. A small NLL change would justify a GPU rollout test; decoding accuracy alone is not enough to accept the edit.

```python
joint194 = joint_decoders193[0.0001]
ne_direction194 = np.asarray(as_np184(directions_ne187[0.1]), dtype=np.float64)
old_b194 = as_np184(submitted193["core.b_in.weight"]).astype(np.float64)

preserved_b194 = (
    old_b194 - np.asarray(delta_b187, dtype=np.float64)
    + np.outer(np.asarray(ins_direction187, dtype=np.float64), joint194["ins"])
)

candidates194 = {}
results194 = {}

def factual_eval194(weights):
    states, signals = replay_factual125(weights)
    losses, probabilities = cpu_factual129(states, weights, draws=4)
    losses = np.asarray(losses)
    assert np.isfinite(states).all() and np.isfinite(losses).all()
    return states, losses, probabilities

try:
    baseline_states194, baseline_losses194, baseline_probs194 = factual_eval194(submitted193)
    print("ORIGINAL NLL", float(np.mean(nll_original129)))
    print("SUBMITTED NLL", float(np.mean(baseline_losses194)),
          "delta", float(np.mean(baseline_losses194 - nll_original129)))

    for scale in (0.0, 0.5, 1.0, 1.5, 2.0):
        weights = dict(submitted193)
        weights.update(dual_candidate193)
        weights["core.b_in.weight"] = torch.as_tensor(
            preserved_b194 + scale * np.outer(ne_direction194, joint194["ne"]),
            dtype=torch.float32
        )
        states, losses, probabilities = factual_eval194(weights)
        candidates194[scale] = weights
        event_deltas = {}
        for event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
            mask = np.asarray(events129) == event
            if mask.any():
                event_deltas[event] = float(np.mean(
                    losses[mask] - np.asarray(nll_original129)[mask]
                ))
        results194[scale] = {
            "losses": losses,
            "mean_nll": float(losses.mean()),
            "delta_original": float(np.mean(losses - nll_original129)),
            "delta_submitted": float(np.mean(losses - baseline_losses194)),
            "event_deltas": event_deltas
        }
        print("NE scale", scale,
              "NLL", round(float(losses.mean()), 7),
              "delta original", round(results194[scale]["delta_original"], 7),
              "delta submitted", round(results194[scale]["delta_submitted"], 7),
              "event deltas", {k: round(v, 6) for k, v in event_deltas.items()})
finally:
    rW = dict(submitted193)

print("\nRETAINED IDENTIFICATION OBJECTS")
for name, value in list(globals().items()):
    if name.endswith(("137", "154")) and isinstance(value, dict):
        print(name, "keys", list(value.keys())[:12])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 194\n"
        "Run-193 amplitude-20 dual channel with ridge .0001 E decoding had "
        "NE R2 .980, insulin-preservation RMSE .02845 and held dose slopes "
        ".867/.929. Tested unsaved MAP-selective NE response scales 0/.5/1/1.5/2 "
        "with training factual replay; original/submitted baselines and per-event "
        "NLL deltas retained in results194. CPU restored; no saved weights changed. "
        "GPU factual and free-rollout validation are still required.\n"
    )
```