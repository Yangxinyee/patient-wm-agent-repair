The state request returned 4,356 bundle states, enough to examine dose encoding and the reading head locally. No weights have changed yet. Next I’ll estimate the rollout targets from the observed dose schedules and inspect the head inputs. This will show how much correction is needed and whether the model’s dose signal can support it without large factual changes.

```python
import torch
import torch.nn.functional as F
import json
from pathlib import Path
torch.set_num_threads(8)
torch.set_grad_enabled(False)

W0 = {n: torch.as_tensor(load_original(n), dtype=torch.float32).clone() for n in NAMES}
S0 = dict(np.load("/work/fwd_3/states_original.npz"))
print("State arrays:", {k: v.shape for k, v in S0.items()})
state_df = pd.DataFrame({k: S0[k] for k in
                         ["stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms"]})
state_df["state_row"] = np.arange(len(state_df))
print("State sample:\n", state_df.head(10).to_string(index=False))
print("Factual IDs match state IDs:",
      set(factual0.stay_id) == set(state_df.stay_id))
print("Actual factual IDs:", sorted(factual0.stay_id.unique().tolist())[:12])
print("VALID_IDS:", VALID_IDS[:12])
print("Rollout columns:", baseline_rollout.columns.tolist())
print(baseline_rollout.head(4).to_string(index=False))

print("\n=== LAST-REGISTER WEIGHT DIAGNOSTICS ===")
A = W0["last_numeric.weight"].numpy()
B = W0["embed.numeric.weight"].numpy()
cosine = A.T @ B / (np.linalg.norm(A, axis=0)[:, None] *
                    np.linalg.norm(B, axis=0)[None, :] + 1e-12)
print("last_numeric column norms:", np.linalg.norm(A, axis=0).round(3))
print("last_numeric vs embed.numeric cosine:\n", cosine.round(2))
for name in ["last_status.weight", "query_norm.weight", "query_norm.bias",
             "encode_action.norm.weight", "z_prior.2.bias", "core.flow.a"]:
    vv = W0[name].numpy()
    print(name, "range", [float(vv.min()), float(vv.max())],
          "norm", float(np.linalg.norm(vv)))
print("Core prelude/coda tensor names:",
      [n for n in NAMES if n.startswith(("core.prelude", "core.coda"))])

def schedule_readout(beta, phi, drug, outcome, source_panel=tpanel):
    sub = source_panel[source_panel.stay_id.isin(state_df.stay_id.unique())].copy()
    results = []
    for sid, g0 in sub.groupby("stay_id", sort=False):
        g0 = g0.set_index("t")
        delta, exposed = 0.0, False
        for hour in range(12, 24):
            dose0 = float(g0.loc[hour, drug]) if hour in g0.index else 0.0
            exposed |= dose0 > 0
            delta = phi * delta + beta * dose0
            if exposed and hour + 1 in g0.index:
                results.append((sid, hour + 1, delta,
                                pd.notna(g0.loc[hour + 1, outcome])))
    r0 = pd.DataFrame(results, columns=["stay_id", "hour", "delta", "recorded"])
    r1 = r0[r0.recorded]
    return {"all_hours": float(r0.delta.mean()),
            "recorded_hours": float(r1.delta.mean()),
            "stay_mean": float(r1.groupby("stay_id").delta.mean().mean()),
            "n_recorded": len(r1), "n_stays": r1.stay_id.nunique()}

print("\n=== PROVISIONAL STRUCTURAL ROLLOUT TARGETS: DOUBLE DOSE ===")
for phi0 in [.80, float(dyn_fit.x[1]), .92]:
    print("insulin", phi0,
          schedule_readout(float(dyn_fit.x[0]), phi0, "insulin", "glucose_finger"))
for phi0 in [.70, .85, .95]:
    print("norepinephrine", phi0,
          schedule_readout(NE_DML, phi0, "norepinephrine", "map_arterial"))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 10: states and target scale\n"
            "Loaded original 4356 bundle states for tuning stays 2020000000..99 "
            "and all original tensors. Computed provisional fixed-dose schedule responses "
            "over several persistence values; these are statistical targets, not hidden truth. "
            "No weights edited.\nPython runs: 10. Forward requests: 3.\n")
```