The unit audit gives norepinephrine dose slopes of about 0.116 for doubling and 0.050 for halving; these replace my earlier interpretation of the diagnostic metrics. I’ll fit several balanced, admission-null detectors and compare factual leakage per unit of dose sensitivity. A candidate is promising only if both signs are correct and its MAP direction has little glucose cross-effect.

```python
from scipy.linalg import solve

# Raw prelude features are converted to the same standardized coordinates
# in which null57 enforces exact admission-state protection.
F81 = (slot_features47[fit_rows57].reshape(-1, 192) / slot_scale47) @ null57
Gf81 = F81.T @ F81 / len(F81)
D81 = {}
Y81 = {}
for key81 in slot_deltas47:
    D81[key81] = (
        array80(slot_deltas47[key81]).reshape(-1, 192) / slot_scale47
    ) @ null57
    Y81[key81] = np.repeat(array80(targets57[key81])[:, 1], 4)

train_slots81 = np.repeat(fit_drug57, 4)
ne_fits81 = {}
ne_metrics81 = []
for half_weight81 in [1.0, 4.0, 10.0]:
    weights81 = {
        ("norepinephrine", 2.0): 1.0,
        ("norepinephrine", 0.5): half_weight81,
        ("insulin", 2.0): 1.0,
        ("insulin", 0.5): 4.0,
    }
    Gc81 = np.zeros((188, 188))
    Rc81 = np.zeros(188)
    for key81, weight81 in weights81.items():
        X81 = D81[key81][train_slots81]
        y81 = Y81[key81][train_slots81]
        Gc81 += weight81 * (X81.T @ X81) / len(X81)
        Rc81 += weight81 * (X81.T @ y81) / len(X81)
    for anchor81 in [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        theta81 = solve(
            Gc81 + anchor81 * Gf81 + 1e-3 * np.eye(188),
            Rc81, assume_a="pos"
        )
        feature81 = (null57 @ theta81) / slot_scale47
        ne_fits81[(half_weight81, anchor81)] = feature81
        factual81 = slot_features47[~fit_rows57] @ feature81
        metric81 = {
            "half_weight": half_weight81,
            "anchor": anchor81,
            "factual_rms": float(np.sqrt(np.mean(factual81.mean(axis=1)**2))),
            "norm": float(np.linalg.norm(feature81)),
            "admission_max": float(np.max(np.abs(initial_matrix56 @ feature81))),
        }
        for key81 in slot_deltas47:
            p81 = (array80(slot_deltas47[key81]) @ feature81).mean(axis=1)
            if key81[0] == "norepinephrine":
                y81 = array80(targets57[key81])[:, 1]
                mask81 = (~fit_drug57) & (np.abs(y81) > 1e-8)
                metric81["slope_" + str(key81[1])] = float(
                    np.dot(p81[mask81], y81[mask81]) /
                    np.dot(y81[mask81], y81[mask81])
                )
            else:
                metric81["insulin_rms_" + str(key81[1])] = float(
                    np.sqrt(np.mean(p81[~fit_drug57]**2))
                )
        metric81["leak_per_x2_slope"] = (
            metric81["factual_rms"] / abs(metric81["slope_2.0"])
        )
        metric81["half_to_double_slope"] = (
            metric81["slope_0.5"] / metric81["slope_2.0"]
        )
        ne_metrics81.append(metric81)

ne_table81 = pd.DataFrame(ne_metrics81)
print("=== BALANCED NE DETECTOR GRID ===")
print(ne_table81.round(5).to_string(index=False))

print("\n=== FORWARD-ONLY OUTPUT DIRECTION CHECK ===")
ne_direction81 = np.asarray(bin_directions46["norepinephrine"], dtype=np.float32)
gain81 = torch.nn.functional.softplus(W0["core.delta_in"])
step81 = 0.1
offset81 = torch.from_numpy(ne_direction81)[None, None, :] * gain81[None, None, :]
hp81, op81 = coda46(hnew46 + step81 * offset81)
hm81, om81 = coda46(hnew46 - step81 * offset81)
response81 = array80(
    (next_means45(hp81, op81) - next_means45(hm81, om81)) / (2 * step81)
)
last_centres81 = array80(centres45)[:, -1]
print("last-bin centres:", np.unique(last_centres81).tolist())
for centre81 in np.unique(last_centres81):
    mask81 = last_centres81 == centre81
    print("centre", float(centre81), "n", int(mask81.sum()),
          "mean response", float(response81[mask81].mean()),
          "relevant mean",
          float(response81[mask81 & relevant46].mean())
          if np.any(mask81 & relevant46) else None)

np.savez("/work/ne_detector81.npz",
         **{f"h{h81:g}_a{a81:g}": f81
            for (h81, a81), f81 in ne_fits81.items()})
with open("/work/notes.md", "a") as file81:
    file81.write(
        "\n## Run 81: independent balanced NE feature fits\n"
        "Raw-unit held-out slopes of the old NE feature are .11648 (double) and "
        ".05022 (half); earlier keyed diagnostics were not these slopes. "
        "Fitted closed-form NE detectors with factual penalties, exact admission "
        "null constraints, insulin-contrast guards, and three halving weights. "
        "Forward-only checked the MAP output direction. No weights changed. "
        "Python runs: 81; forwards: 10.\n"
    )
```