The recovered kernels match the original record estimates. The target discrepancy may therefore come from the selected stays or averaging, rather than different drug-effect evidence. I will compare both conventions on identical rows and inspect the previously validated stronger-insulin edit. If it has a safer factual margin and better-aligned insulin effects, it merits a final rollout comparison.

```python
import os, json
import numpy as np
import pandas as pd

audit298_rows = pd.read_csv("/work/final_rollout400_record_comparison.csv")
audit298_rows = audit298_rows.loc[audit298_rows["reached"].astype(bool)].copy()
audit298_rates = r132_rates
audit298_kernels = final_empirical_impulses
audit298_edit_info = {
    "insulin_x2": ("insulin", 1.0),
    "insulin_half": ("insulin", -0.5),
    "ne_x2": ("norepinephrine", 1.0),
    "ne_half": ("norepinephrine", -0.5),
}

def audit298_target(stay, hour, edit):
    drug, change = audit298_edit_info[edit]
    result = 0.0
    for lag, coefficient in enumerate(audit298_kernels[drug]):
        dose_hour = int(hour) - 1 - lag
        if dose_hour >= 12:
            result += coefficient * float(
                audit298_rates.get((int(stay), dose_hour, drug), 0.0)
            )
    return change * result

audit298_rows["recomputed_target"] = [
    audit298_target(row.stay_id, row.hour, row.edit)
    for row in audit298_rows.itertuples(index=False)
]
print("=== SAME-ROW CONVOLUTION CHECK ===")
print("Maximum discrepancy:", float(np.max(np.abs(
    audit298_rows["recomputed_target"]
    - audit298_rows["empirical_six_lag_target"]
))))

audit298_base_id = int(
    audit298_rates.index.get_level_values("stay_id").min()
)
audit298_tables = []
for label, cutoff in (("first80", 80), ("first100", 100), ("all400", 400)):
    subset = audit298_rows.loc[
        audit298_rows["stay_id"] < audit298_base_id + cutoff
    ]
    for edit, group in subset.groupby("edit"):
        per_stay = group.groupby("stay_id")[
            ["shift", "recomputed_target"]
        ].mean()
        audit298_tables.append({
            "cohort": label,
            "edit": edit,
            "n_stays": len(per_stay),
            "n_hours": len(group),
            "hour_weighted_prediction": group["shift"].mean(),
            "hour_weighted_target": group["recomputed_target"].mean(),
            "stay_weighted_prediction": per_stay["shift"].mean(),
            "stay_weighted_target": per_stay["recomputed_target"].mean(),
        })
print(pd.DataFrame(audit298_tables).round(6).to_string(index=False))

print("\n=== EXACT OLDER TARGET COHORT ===")
print("Retained run-132 targets:", r132_targets)
print(
    "Retained rows:",
    len(r132_rows),
    "stays:", r132_rows["stay_id"].nunique(),
    "ID range:",
    int(r132_rows["stay_id"].min()),
    int(r132_rows["stay_id"].max()),
)
audit298_old = r132_rows.copy()
audit298_old["recomputed_target"] = [
    audit298_target(row.stay_id, row.hour, row.edit)
    for row in audit298_old.itertuples(index=False)
]
print(audit298_old[
    ["target_short", "target_long", "recomputed_target"]
].mean().to_string())
print("Old short-target maximum discrepancy:", float(np.max(np.abs(
    audit298_old["target_short"] - audit298_old["recomputed_target"]
))))

def audit298_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

audit298_current = audit298_numpy(load("core.b_in.weight"))
audit298_original = audit298_numpy(load_original("core.b_in.weight"))
audit298_archives = {}
print("\n=== CANDIDATE ARCHIVES ===")
print("Currently edited:", edited())
for filename in (
    "validated_current400.npz",
    "margin_validated400.npz",
    "balanced_validated400.npz",
):
    with np.load("/work/" + filename, allow_pickle=False) as archive:
        audit298_archives[filename] = {}
        print(filename, "keys:", archive.files)
        for key in archive.files:
            try:
                value = archive[key]
            except ValueError:
                print(key, "(object metadata omitted)")
                continue
            audit298_archives[filename][key] = value.copy()
            if value.shape == audit298_current.shape:
                print(
                    key,
                    "norm:", float(np.linalg.norm(value)),
                    "distance from current:", float(np.linalg.norm(
                        value - audit298_current
                    )),
                    "distance from original:", float(np.linalg.norm(
                        value - audit298_original
                    )),
                    "max difference from current:", float(np.max(np.abs(
                        value - audit298_current
                    ))),
                )
            elif value.size <= 12:
                print(key, value.tolist())
            else:
                print(key, "shape:", value.shape)

print("\n=== FINAL CANDIDATE FACTUAL CHECK ===")
audit298_factual = pd.read_csv("/work/margin_paired400.csv")
print("Rows:", len(audit298_factual), "columns:", list(audit298_factual.columns))
print(audit298_factual[
    ["original_nll", "edited_nll", "delta_nll"]
].mean().to_string())

print("\n=== AVAILABLE FORWARD METADATA ===")
for folder in ("fwd_37", "fwd_38", "fwd_40", "fwd_41", "fwd_42"):
    directory = "/work/" + folder
    filenames = os.listdir(directory)
    print(folder, filenames)
    for filename in filenames:
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename)) as file:
                metadata = json.load(file)
            print(filename, str(metadata)[:700])

with open("/work/notes.md", "a") as file:
    file.write(
        "\n## Run 298: same-row target and archived-candidate audit\n"
        "- Recomputed six-lag physical-dose targets on identical reached rows; "
        "compared first-80, first-100 and full-400 cohorts under hour and stay weighting.\n"
        "- Inspected exact archived stronger-insulin and margin candidates; "
        "no weights changed.\n"
        "- Current factual delta remains +0.003384797 nats per reading. "
        "The previously validated glucose-1.15/MAP-0.5 candidate had "
        "+0.003163118, with stay-bootstrap 97.5% bound +0.004535967.\n"
        "- Python runs: 298; forward passes: 44.\n"
    )
```