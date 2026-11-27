The unit audit confirms that the contrasts use actual recorded rates. The old summary helper failed, so I’ll calculate diagnostics explicitly. I’ll rebalance insulin’s doubling and halving sensitivities while preserving the admission-null constraint and suppressing norepinephrine cross-response, then test a stronger reversible candidate. Factual validation will determine whether that added sensitivity is affordable.

```python
import numpy as np
from pathlib import Path

keys63 = list(slot_deltas47)
D63 = {k: np.asarray(slot_deltas47[k], dtype=np.float64).mean(axis=1)
       for k in keys63}
Y63 = {k: np.asarray(targets57[k], dtype=np.float64) for k in keys63}
A63 = []
for k63 in keys63:
    col63 = 0 if k63[0] == "insulin" else 1
    y63 = Y63[k63][:, col63]
    active63 = np.abs(y63) > 1e-10
    A63.append(D63[k63][active63].mean(axis=0) / y63[active63].mean())
A63 = np.stack(A63)

base_input63 = np.asarray(candidate_input58, dtype=np.float64)
base_slopes63 = A63 @ base_input63
desired_slopes63 = np.array([base_slopes63[0], base_slopes63[0], 0.0, 0.0])

basis63 = np.asarray(null57, dtype=np.float64) / np.asarray(scale57)[:, None]
precision63 = (
    np.asarray(null57).T
    @ (np.asarray(Gc57) + 10.0*np.asarray(Gf57) + 0.1*np.eye(192))
    @ np.asarray(null57)
)
base_coordinates63 = np.linalg.lstsq(basis63, base_input63, rcond=None)[0]
constraints63 = A63 @ basis63
inverse_constraints63 = np.linalg.solve(precision63, constraints63.T)
constraint_cov63 = constraints63 @ inverse_constraints63
balanced_coordinates63 = (
    base_coordinates63
    + inverse_constraints63
    @ np.linalg.solve(constraint_cov63,
                      desired_slopes63 - constraints63 @ base_coordinates63)
)
balanced_input63 = basis63 @ balanced_coordinates63

heldout_raw63 = np.asarray(raw_E57)[~np.asarray(fit_rows57, dtype=bool)]
base_factual63 = heldout_raw63 @ base_input63
balanced_factual63 = heldout_raw63 @ balanced_input63
base_rms63 = float(np.sqrt(np.mean(base_factual63**2)))
balanced_rms63 = float(np.sqrt(np.mean(balanced_factual63**2)))
initial_rms63 = float(np.sqrt(np.mean(
    (np.asarray(initial_matrix56) @ balanced_input63)**2)))

print("contrast keys:", keys63)
print("base mean dose sensitivities:", base_slopes63)
print("balanced mean dose sensitivities:", A63 @ balanced_input63)
print("held-out factual RMS, base/balanced:", base_rms63, balanced_rms63)
print("admission RMS:", initial_rms63)
print("input norms, base/balanced:",
      np.linalg.norm(base_input63), np.linalg.norm(balanced_input63))
for k63 in keys63[:2]:
    col63 = 0
    y63 = Y63[k63][:, col63]
    active63 = (np.abs(y63) > 1e-10) & ~np.asarray(fit_drug57, dtype=bool)
    prediction63 = D63[k63][active63] @ balanced_input63
    print("held-out", k63,
          "mean sensitivity", prediction63.mean()/y63[active63].mean(),
          "OLS sensitivity", prediction63 @ y63[active63]
          / (y63[active63] @ y63[active63]))

strength63 = 1.9
candidate_delta63 = (
    trial_strength47 * strength_multiplier58 * strength63
    * np.outer(np.asarray(direction58), balanced_input63)
).astype(np.float32)
installed63 = (
    np.isfinite(candidate_delta63).all()
    and initial_rms63 < 1e-5
    and balanced_rms63 < 2.5*base_rms63
)
if installed63:
    save_edit("core.b_in.weight",
              load_original("core.b_in.weight") + candidate_delta63)
    np.savez("/work/insulin_candidate63.npz",
             delta=candidate_delta63, feature=balanced_input63,
             direction=np.asarray(direction58), strength=strength63)
print("installed:", installed63, "delta norm:", np.linalg.norm(candidate_delta63))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 62–63: balanced insulin candidate\n"
            "Run 62 verified raw-rate contrast units; old feature_metrics helper "
            "failed, so diagnostics were recomputed explicitly. "
            "Closed-form precision-weighted correction enforces equal mean "
            "insulin x2/half sensitivity, zero mean NE cross-response, and "
            "exact admission nullspace. "
            f"Base slopes={base_slopes63.tolist()}, balanced="
            f"{(A63 @ balanced_input63).tolist()}; heldout factual RMS "
            f"{base_rms63:.6f}/{balanced_rms63:.6f}; initial RMS "
            f"{initial_rms63:.3g}. Strength relative to candidate58={strength63}; "
            f"installed={installed63}. Not accepted before GPU validation. "
            "Python runs: 63; forwards: 7.\n")
```
