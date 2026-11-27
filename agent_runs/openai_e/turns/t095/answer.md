The grid shows that simply upweighting halving can reverse the doubling response. Also, the old MAP direction is glucose-neutral only in selected contexts. I’ll strengthen the insulin-contrast guards and compute a MAP direction across all available calibration contexts. This approach fails if selectivity requires excessive factual leakage or unstable state changes.

```python
# Audit whether the retained Jacobian includes the recurrent gain.
pred82 = np.asarray(jac_bin46) @ ne_direction81
jac_error82 = np.linalg.norm(pred82 - response81) / max(np.linalg.norm(response81), 1e-8)
print("Retained Jacobian relative error:", float(jac_error82))

if jac_error82 < 0.02:
    J82 = np.asarray(jac_bin46, dtype=np.float64).copy()
else:
    J82 = np.empty((len(src45), 192), dtype=np.float64)
    eps82 = 0.05
    for dim82 in range(192):
        offset82 = torch.zeros_like(hnew46)
        offset82[:, :, dim82] = gain81[dim82]
        hp82, op82 = coda46(hnew46 + eps82 * offset82)
        hm82, om82 = coda46(hnew46 - eps82 * offset82)
        J82[:, dim82] = array80(
            (next_means45(hp82, op82) - next_means45(hm82, om82)) / (2 * eps82)
        )

map_mask82 = last_centres81 == 101.0
glucose_mask82 = ~map_mask82
desired82 = map_mask82.astype(float)
directions82 = {}
print("\n=== MULTI-CONTEXT MAP DIRECTIONS ===")
for ridge82 in [0.001, 0.01, 0.1, 1.0]:
    direction82 = J82.T @ solve(
        J82 @ J82.T + ridge82 * np.eye(len(J82)),
        desired82, assume_a="pos"
    )
    normalization82 = np.mean((J82 @ direction82)[map_mask82 & relevant46])
    direction82 /= normalization82
    directions82[ridge82] = direction82
    response82 = J82 @ direction82
    print("ridge", ridge82, "norm", np.linalg.norm(direction82),
          "glucose RMS", np.sqrt(np.mean(response82[glucose_mask82]**2)),
          "MAP mean/RMS error",
          response82[map_mask82].mean(),
          np.sqrt(np.mean((response82[map_mask82] - 1)**2)))

# Fit stronger non-target-drug guards. No model weights are optimized.
ne_fits82 = {}
records82 = []
for cross_weight82 in [1.0, 10.0, 100.0]:
    for half_weight82 in [1.0, 1.15, 1.3]:
        weights82 = {
            ("norepinephrine", 2.0): 1.0,
            ("norepinephrine", 0.5): half_weight82,
            ("insulin", 2.0): cross_weight82,
            ("insulin", 0.5): 4 * cross_weight82,
        }
        Gc82 = np.zeros((188, 188))
        Rc82 = np.zeros(188)
        for key82, weight82 in weights82.items():
            X82 = D81[key82][train_slots81]
            y82 = Y81[key82][train_slots81]
            Gc82 += weight82 * (X82.T @ X82) / len(X82)
            Rc82 += weight82 * (X82.T @ y82) / len(X82)
        for anchor82 in [300.0, 1000.0, 3000.0]:
            theta82 = solve(
                Gc82 + anchor82 * Gf81 + 1e-3 * np.eye(188),
                Rc82, assume_a="pos"
            )
            feature82 = (null57 @ theta82) / slot_scale47
            keyfit82 = (cross_weight82, half_weight82, anchor82)
            ne_fits82[keyfit82] = feature82
            metric82 = {"cross": cross_weight82, "half_w": half_weight82,
                        "anchor": anchor82, "norm": np.linalg.norm(feature82)}
            factual82 = (slot_features47[~fit_rows57] @ feature82).mean(axis=1)
            metric82["factual_rms"] = np.sqrt(np.mean(factual82**2))
            for key82 in slot_deltas47:
                p82 = (array80(slot_deltas47[key82]) @ feature82).mean(axis=1)
                if key82[0] == "norepinephrine":
                    y82 = array80(targets57[key82])[:, 1]
                    mask82 = (~fit_drug57) & (np.abs(y82) > 1e-8)
                    metric82["s" + str(key82[1])] = (
                        np.dot(p82[mask82], y82[mask82]) /
                        np.dot(y82[mask82], y82[mask82])
                    )
                else:
                    metric82["cross_rms_" + str(key82[1])] = (
                        np.sqrt(np.mean(p82[~fit_drug57]**2))
                    )
            s82 = metric82["s2.0"]
            metric82["unit_factual"] = metric82["factual_rms"] / max(abs(s82), 1e-8)
            metric82["unit_cross"] = (
                metric82["cross_rms_2.0"] + metric82["cross_rms_0.5"]
            ) / max(abs(s82), 1e-8)
            metric82["half_ratio"] = metric82["s0.5"] / s82
            metric82["score"] = (
                metric82["unit_factual"] + .2 * metric82["unit_cross"] +
                .03 * abs(metric82["half_ratio"] - .55)
            ) if s82 > 0 and metric82["s0.5"] > 0 else np.inf
            records82.append(metric82)

ne_table82 = pd.DataFrame(records82).sort_values("score")
print("\n=== SELECTIVE NE FITS, BEST 12 ===")
print(ne_table82.head(12).round(5).to_string(index=False))
best82 = ne_table82.iloc[0]
chosen_key82 = (float(best82["cross"]), float(best82["half_w"]),
                float(best82["anchor"]))
print("Selected detector:", chosen_key82)
ne_feature82 = ne_fits82[chosen_key82].copy()
ne_slope82 = float(best82["s2.0"])
ne_direction82 = directions82[0.01].copy()

# Independent central-difference confirmation of the selected direction.
offset82 = torch.from_numpy(ne_direction82.astype(np.float32))[None, None, :] * gain81[None, None, :]
hp82, op82 = coda46(hnew46 + .1 * offset82)
hm82, om82 = coda46(hnew46 - .1 * offset82)
check82 = array80(
    (next_means45(hp82, op82) - next_means45(hm82, om82)) / .2
)
print("Selected direction finite-difference glucose RMS:",
      float(np.sqrt(np.mean(check82[glucose_mask82]**2))))
print("Selected direction finite-difference MAP mean:",
      float(check82[map_mask82].mean()))

np.savez("/work/ne_candidate82.npz", feature=ne_feature82,
         direction=ne_direction82, x2_slope=ne_slope82,
         choice=np.asarray(chosen_key82), jacobian=J82)
with open("/work/notes.md", "a") as file82:
    file82.write(
        "\n## Run 82: selective NE detector and multi-context MAP direction\n"
        "Heavy halving weights can reverse doubling sensitivity. Strengthened "
        "insulin contrast guards and factual penalties; fitted a broader "
        "glucose-neutral MAP output direction using forward differences only. "
        f"Selected detector={chosen_key82}, metrics={best82.to_dict()}. "
        "No weights changed. Python runs: 82; forwards: 10.\n"
    )
```