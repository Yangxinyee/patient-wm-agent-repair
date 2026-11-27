The glucose edit added only 0.00184 nats on the second batch, leaving more room than the first batch suggested. I’ll now fit the MAP feature against record-estimated effects across all five dose multipliers, while penalizing factual probability changes. A useful candidate should improve both doubled and halved responses on local cases held out from the fit.

```python
import numpy as np, pandas as pd, torch

r276_factual_stats = []
for k in [24, 25]:
    o = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    e = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    assert len(o) == len(e)
    r276_factual_stats.append((len(o), float(e.nll.mean()-o.nll.mean())))
print("Glucose-only factual batches:", r276_factual_stats)
print("Pooled factual delta:",
      sum(n*d for n,d in r276_factual_stats) /
      sum(n for n,d in r276_factual_stats))

r276_n = len(indices151)
r276_fitmask = np.arange(r276_n) % 2 == 0
r276_valmask = ~r276_fitmask
r276_fmean = np242(r275_mean0)
r276_Jf = r275_Jfeature_mean
r276_features = {}
r276_base_delta = {}
r276_rates = {}

for key, Ypart in zip(r245_keys, r245_Yparts):
    case = r272_cases[key]
    pre = r274_pre(case, r272_glucose)
    means = np242(r244_read_pre(pre, indices151))
    Jslot = np.empty((r276_n, 4, 4), dtype=np.float64)
    for slot in range(4):
        step = torch.zeros_like(pre)
        step[:, slot, :] = torch.as_tensor(
            r275_eps*r275_direction, dtype=torch.float32
        )
        plus = np242(r244_read_pre(pre+step, indices151))
        minus = np242(r244_read_pre(pre-step, indices151))
        Jslot[:, :, slot] = (plus-minus)/(2*r275_eps)
    Jfeature = np.einsum(
        "brs,bsi->bri", Jslot, np242(case["E"]).astype(np.float64)
    )
    r276_features[key] = Jfeature-r276_Jf
    r276_base_delta[key] = means-r276_fmean
    r276_rates[key] = np242(Ypart).reshape(r276_n, 4, 2).mean(axis=1)

r276_coefficients = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 3.05259229],
    [0.0, 2.52816442]
])
r276_Aparts, r276_yparts = [], []
for key in r245_keys:
    A = r276_features[key][r276_fitmask].mean(axis=0)
    base = r276_base_delta[key][r276_fitmask].mean(axis=0)
    if key[0] == "norepinephrine":
        desired = r276_coefficients @ r276_rates[key][r276_fitmask].mean(axis=0)
        y = desired-base
        importance = 1.0
    else:
        y = np.array([0.0, 0.0, -base[2], -base[3]])
        importance = 0.5
    r276_Aparts.append(np.sqrt(importance)*A)
    r276_yparts.append(np.sqrt(importance)*y)

# Keep the average factual shift centered, in addition to the KL penalty.
r276_Aparts.append(2.0*r276_Jf[r276_fitmask].mean(axis=0))
r276_yparts.append(np.zeros(4))
r276_A = np.concatenate(r276_Aparts, axis=0)
r276_y = np.concatenate(r276_yparts)

r276_design_by_case = r275_design.reshape(r276_n, -1, 192)
r276_Dfit = r276_design_by_case[r276_fitmask].reshape(-1,192)
r276_Dval = r276_design_by_case[r276_valmask].reshape(-1,192)
r276_Ffit = r276_Dfit.T @ r276_Dfit * r276_n/r276_fitmask.sum()
r276_Fval = r276_Dval.T @ r276_Dval * r276_n/r276_valmask.sum()
r276_Eflat = r275_E.reshape(-1,192)
r276_cov = r276_Eflat.T @ r276_Eflat / len(r276_Eflat)
r276_ridge = 1e-6*np.trace(r276_Ffit)/192

print("Record-based local NE x2 target:",
      (r276_coefficients @
       r276_rates[("norepinephrine",2.0)].mean(axis=0)).round(6).tolist())

r276_trials = {}
for slot_penalty in [0.0, 0.0001, 0.001, 0.01]:
    M = r276_Ffit + slot_penalty*r276_cov + r276_ridge*np.eye(192)
    B = np.linalg.solve(M, r276_A.T)
    G = r276_A @ B
    for softness in [0.05, 0.5, 5.0]:
        v = B @ np.linalg.solve(G+softness*np.eye(len(G)), r276_y)
        raw_cost = max(
            0.5*float(v @ r276_Ffit @ v),
            0.5*float(v @ r276_Fval @ v)
        )
        scale = min(1.0, np.sqrt(0.0012/max(raw_cost, 1e-15)))
        v = scale*v
        report = {
            "feature_norm": float(np.linalg.norm(v)),
            "quadratic_KL_fit": float(0.5*v @ r276_Ffit @ v),
            "quadratic_KL_validation": float(0.5*v @ r276_Fval @ v),
            "slot_feature_RMS": float(np.sqrt(max(v @ r276_cov @ v,0))),
            "scale": float(scale)
        }
        for mult in [0.0, 0.5, 1.5, 2.0, 3.0]:
            key = ("norepinephrine",mult)
            predicted = (
                r276_base_delta[key][r276_valmask].mean(axis=0) +
                r276_features[key][r276_valmask].mean(axis=0) @ v
            )
            report[f"NE_{mult}_validation_shift"] = predicted.round(5).tolist()
        r276_trials[(slot_penalty,softness)] = {"v":v, "summary":report}
        print("penalty/softness", slot_penalty, softness, report)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 276: slot-aware dose-feature regression\n"
        "- Glucose-only strength 1.15 passed a second 100-stay factual batch; computed pooled exact delta from forwards 24 and 25.\n"
        "- Built dose-contrast features by scalar-slot finite differences, for x0/.5/1.5/2/3 of both drugs.\n"
        "- Fit small linear dose-feature models using record-estimated MAP effects, factual probability distortion, and optional slot-energy penalties.\n"
        "- Alternating local cases held out from fitting; candidate quadratic KL capped at 0.0012 on both halves.\n"
        "- No weight changes, backpropagation, or weight optimizer.\n"
        "- Python runs: 276; forward passes: 25.\n"
    )
```