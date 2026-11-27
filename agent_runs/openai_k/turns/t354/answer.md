All four batches now show a stronger insulin response, while norepinephrine remains under-responsive. I’ll calculate the exact pooled shifts and compare them with record-based convolution estimates on the same stay-hours. Agreement would support insulin calibration; a mismatch would reveal remaining bias rather than justify claiming a complete repair.

```python
from pathlib import Path
import numpy as np, pandas as pd

final_rollout_folders = ["fwd_42", "fwd_43", "fwd_44", "fwd_45"]
final_rollouts = {}
for _model in ["original", "edited"]:
    _d = pd.concat([
        pd.read_csv(f"/work/{f}/dose_edit_{_model}.csv")
        for f in final_rollout_folders
    ], ignore_index=True)
    _d = _d.loc[_d["reached"].astype(str).str.lower().eq("true")].copy()
    final_rollouts[_model] = _d
    print("POOLED", _model)
    print(_d.groupby("edit")["shift"].agg(["mean", "count"]).to_string())

# These impulse coefficients were estimated from the supplied training records
# in run 6; they are identification-based estimates, not counterfactual truth.
final_empirical_impulses = {
    "insulin": np.array([-2.07283356, -2.23466817, -2.14515517,
                         -1.43875875, -1.28052947, -0.90473374]),
    "norepinephrine": np.array([2.95159917, 2.49611187, 1.62799136,
                                1.02869814, 0.34279884, 0.21392442])
}
_final_records = pd.read_csv("/data/tuning.csv")
_final_drugs = _final_records[
    _final_records["event"].isin(["insulin", "norepinephrine"])
].copy()
_final_drugs["dose_hour"] = np.floor(_final_drugs["hour"] + 1e-6).astype(int)
_final_rates = _final_drugs.groupby(
    ["stay_id", "event", "dose_hour"]
)["rate"].sum().to_dict()

def final_convolution(stay, hour, drug, coefficients):
    h = int(round(float(hour)))
    return sum(float(coefficients[h - j - 1]) *
               float(_final_rates.get((int(stay), drug, j), 0.0))
               for j in range(12, h)
               if 1 <= h - j <= len(coefficients))

final_target_rows = final_rollouts["edited"].copy()
final_target_rows["empirical_six_lag_target"] = [
    final_convolution(s, h,
                      "insulin" if e.startswith("insulin") else "norepinephrine",
                      final_empirical_impulses[
                          "insulin" if e.startswith("insulin") else "norepinephrine"
                      ]) * (-0.5 if e.endswith("half") else 1.0)
    for s, h, e in final_target_rows[["stay_id", "hour", "edit"]].itertuples(
        index=False, name=None)
]
print("RECORD-ESTIMATED SIX-LAG TARGETS, SAME READOUT ROWS")
print(final_target_rows.groupby("edit")[
    ["shift", "empirical_six_lag_target"]
].mean().to_string())
print("FIRST-BATCH TARGET REPRODUCTION")
print(final_target_rows[final_target_rows.stay_id < 2020000100]
      .groupby("edit")["empirical_six_lag_target"].mean().to_string())

_final_boot_rng = np.random.default_rng(295)
final_rollout_summary = []
for _edit, _d in final_target_rows.groupby("edit"):
    _s = _d.groupby("stay_id")["shift"].agg(["sum", "count"])
    _draw = _final_boot_rng.integers(0, len(_s), size=(5000, len(_s)))
    _boot = (_s["sum"].to_numpy()[_draw].sum(axis=1) /
             _s["count"].to_numpy()[_draw].sum(axis=1))
    final_rollout_summary.append({
        "edit": _edit, "mean_shift": _d["shift"].mean(),
        "stay_bootstrap_low": np.quantile(_boot, 0.025),
        "stay_bootstrap_high": np.quantile(_boot, 0.975),
        "empirical_six_lag_target": _d["empirical_six_lag_target"].mean(),
        "n_stay_hours": len(_d), "n_stays": _d.stay_id.nunique()
    })
final_rollout_summary = pd.DataFrame(final_rollout_summary)
print("ROLLOUT STAY-BOOTSTRAP SUMMARY")
print(final_rollout_summary.to_string(index=False))
final_rollout_summary.to_csv("/work/final_rollout400_summary.csv", index=False)
final_target_rows.to_csv("/work/final_rollout400_record_comparison.csv", index=False)

_final_factual = pd.read_csv("/work/margin_paired400.csv")
print("FACTUAL COLUMN NAMES:", _final_factual.columns.tolist())
print("FACTUAL VALUE-NLL DELTA:",
      _final_factual["delta_nll"].mean(),
      "readings:", len(_final_factual),
      "stays:", _final_factual.stay_id.nunique())
print("CURRENT EDITS:", edited())

with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Runs 294-295: complete unchanged-candidate rollout validation\n"
             "- Run 294 confirmed exact weight equality to margin_validated400.npz, "
             "then stopped on a pandas method/column-name collision; no weights changed.\n"
             "- Forwards 43/44 (folders fwd_44/fwd_45) completed all 400 tuning rollouts.\n"
             "- Run 295 pooled reached stay-hours and compared against six-lag "
             "training-record impulse convolutions. These targets require causal identification "
             "assumptions and omit the longer tail; they are not observed truth.\n"
             "- Saved final_rollout400_summary.csv and final_rollout400_record_comparison.csv.\n"
             "- Python runs: 295; forward passes: 44.\n")
    _f.write(final_rollout_summary.to_string(index=False) + "\n")
```