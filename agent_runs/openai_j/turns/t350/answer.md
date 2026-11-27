All four tuning groups already have factual and rollout results, so no additional pass is needed to complete coverage. The records-based target is roughly −11/+5.5 for insulin and +4/−2 for norepinephrine, with substantial uncertainty—especially for insulin’s long-horizon effect. I’ll pool the paired results using the evaluator’s reached-hour weighting and check factual safety. This will show the candidate’s remaining mismatch without treating those targets as known truth.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import re

root = Path("/work")
dose_dirs = [42, 46, 47, 48]
pairs = []
print("ROLLOUT GROUP AUDIT:")
for k in dose_dirs:
    folder = root / f"fwd_{k}"
    summary = json.loads((folder / "summary.json").read_text())
    print(folder.name, "type", summary.get("type"),
          "split", summary.get("split"), "n_stays", summary.get("n_stays"))
    for key, value in summary.items():
        if "hash" in key.lower():
            print(" ", key, value)
    o = pd.read_csv(folder / "dose_edit_original.csv")
    e = pd.read_csv(folder / "dose_edit_edited.csv")
    keys = ["stay_id", "hour", "edit"]
    p = o.merge(e, on=keys, suffixes=("_original", "_edited"),
                validate="one_to_one")
    assert len(p) == len(o) == len(e)
    def reached_mask(series):
        return series.astype(str).str.lower().isin(["true", "1", "1.0"])
    ro = reached_mask(p["reached_original"])
    re_ = reached_mask(p["reached_edited"])
    assert np.array_equal(ro.to_numpy(), re_.to_numpy())
    p = p.loc[ro].copy()
    p["forward_dir"] = k
    pairs.append(p)

pooled_dose_final = pd.concat(pairs, ignore_index=True)
assert not pooled_dose_final.duplicated(["stay_id", "hour", "edit"]).any()
all_ids = np.asarray(tune_ids_final)
rng = np.random.default_rng(295)
bootstrap_counts = rng.multinomial(
    len(all_ids), np.full(len(all_ids), 1 / len(all_ids)), size=4000
)
rows = []
for edit, df in pooled_dose_final.groupby("edit", sort=False):
    grouped = df.groupby("stay_id").agg(
        n=("shift_edited", "size"),
        so=("shift_original", "sum"),
        se=("shift_edited", "sum"),
    ).reindex(all_ids, fill_value=0)
    denom = bootstrap_counts @ grouped["n"].to_numpy()
    boot = (bootstrap_counts @ grouped["se"].to_numpy()) / denom
    rows.append({
        "edit": edit, "n_stay_hours": len(df),
        "n_reached_stays": df["stay_id"].nunique(),
        "original": df["shift_original"].mean(),
        "edited": df["shift_edited"].mean(),
        "stay_boot_lo": np.quantile(boot, .025),
        "stay_boot_hi": np.quantile(boot, .975),
    })
rollout_summary_final = pd.DataFrame(rows)
print("\nPOOLED REACHED-HOUR READOUTS:")
print(rollout_summary_final.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\nEDITED READOUT BY GROUP:")
print(pooled_dose_final.groupby(["forward_dir", "edit"])["shift_edited"]
      .agg(["mean", "size"]).to_string())

audit = pd.read_csv(root / "final_factual_audit289.csv")
assert audit["stay_id"].nunique() == 400 and len(audit) == 18654
assert not audit.duplicated(["stay_id", "hour", "event"]).any()
assert np.allclose(audit["delta"], audit["edited_nll"] - audit["original_nll"],
                   atol=1e-7)
g = audit.groupby("stay_id").agg(
    n=("delta", "size"), total=("delta", "sum")
).reindex(all_ids, fill_value=0)
boot_nll = ((bootstrap_counts @ g["total"].to_numpy()) /
            (bootstrap_counts @ g["n"].to_numpy()))
factual_summary_final = {
    "n_stays": 400, "n_readings": len(audit),
    "original_nll": float(audit["original_nll"].mean()),
    "edited_nll": float(audit["edited_nll"].mean()),
    "delta_nll": float(audit["delta"].mean()),
    "stay_boot_95": np.quantile(boot_nll, [.025, .975]).tolist(),
}
print("\nPOOLED FACTUAL SAFETY:")
print(json.dumps(factual_summary_final, indent=2))
print("\nFACTUAL DELTA BY SENSOR:")
print(audit.groupby("event")["delta"].agg(["mean", "size"]).to_string())

notes = (root / "notes.md").read_text()
print("\nEARLY RECORD-BASED ESTIMATION NOTES:")
print(notes[:2800])
print("\nCANDIDATE-BUILDER FUNCTIONS:")
print([name for name, value in list(globals().items())
       if callable(value) and re.search(r"build|candidate|calibrat|safe", name, re.I)][:35])

rollout_summary_final.to_csv(root / "final_rollout_audit295.csv", index=False)
(root / "final_validation295.json").write_text(
    json.dumps({"factual": factual_summary_final,
                "rollouts": rollout_summary_final.to_dict("records")}, indent=2)
)
with (root / "notes.md").open("a") as f:
    f.write(
        "\n## Run 295: completed pooled validation\n"
        "- Pooled paired dose rollouts from fwd_42/46/47/48, restricted "
        "to matched reached stay-hours; all four tuning groups covered.\n"
        + rollout_summary_final.to_string(index=False) + "\n"
        + "- Factual audit: " + json.dumps(factual_summary_final) + "\n"
        "- Bootstrap intervals are conditional on candidate selection and "
        "do not establish the simulator's true causal effects.\n"
        "- No weights changed. Counts: 295 Python runs, 46 forward passes.\n"
    )
```