The norepinephrine window-average estimate is weak and uncertain, including zero; it does not establish a persistent dose effect. Together with the earlier short-term checks, that favors retaining the null edit, without proving absence. I’ll audit the tensors and select high-exposure stays for a final rollout stress test. Any nonzero edited shift would reveal a dose-information path the edit missed.

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path

def audit_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

expected_edits = {
    "embed.numeric.weight",
    "embed.bins.dose.weight",
    "embed.bins.rate.weight"
}
assert set(edited()) == expected_edits
nw = audit_numpy(load("embed.numeric.weight"))
ow = audit_numpy(load_original("embed.numeric.weight"))
assert nw.shape[1] == 6
assert np.count_nonzero(nw[:, [2, 4]]) == 0
assert np.array_equal(nw[:, [0, 1]], ow[:, [0, 1]])

audit_results_final = []
for tensor_name in ["embed.bins.dose.weight", "embed.bins.rate.weight"]:
    edited_table = audit_numpy(load(tensor_name))
    original_table = audit_numpy(load_original(tensor_name))
    assert np.array_equal(edited_table[0], original_table[0])
    assert np.array_equal(
        edited_table[1:], np.broadcast_to(edited_table[1], edited_table[1:].shape)
    )
    audit_results_final.append({
        "tensor": tensor_name,
        "shape": list(edited_table.shape),
        "absent_row_unchanged": True,
        "all_present_rows_identical": True
    })
print("=== WEIGHT AUDIT ===")
print(audit_results_final)
print("Numeric dose/rate columns exactly zero; measurement columns unchanged.")

tuning_for_selection = pd.read_csv(
    "/data/tuning.csv", usecols=["stay_id", "hour", "event", "rate"]
)
late_actions = tuning_for_selection[
    tuning_for_selection.event.isin(["insulin", "norepinephrine"])
    & tuning_for_selection.hour.between(12, 23)
]
stress_scores = (
    late_actions.groupby(["stay_id", "event"]).rate.sum()
    .unstack(fill_value=0)
    .reindex(tuning_ids_all, fill_value=0)
)
stress_ne_ids = stress_scores.nlargest(50, "norepinephrine").index.tolist()
stress_ins_ids = (
    stress_scores.drop(index=stress_ne_ids)
    .nlargest(50, "insulin").index.tolist()
)
stress_test_ids = sorted(int(s) for s in stress_ne_ids + stress_ins_ids)
assert len(stress_test_ids) == 100 and len(set(stress_test_ids)) == 100
print("=== HIGH-EXPOSURE STRESS-TEST IDS ===")
print(stress_test_ids)
print("Selected stays with positive late exposure:",
      (stress_scores.loc[stress_test_ids] > 0).sum().to_dict())

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Final candidate audit and stress-test selection, run 35\n")
    f.write(str(audit_results_final) + "\n")
    f.write("Numeric magnitude columns are zero; value columns and absent bin "
            "rows remain original. The current best-supported conclusion is "
            "approximately null dose effects. This is not proof: observational "
            "identification and extrapolation beyond observed rates remain "
            "limitations, and the isolated NE lag-12 signal remains uncertain. "
            "No new effect is introduced solely from that unstable signal.\n")
    f.write("Final rollout test selected by records alone: 50 highest late-NE "
            "exposure stays plus 50 highest late-insulin exposure stays among "
            "the remainder. No weights changed.\n")
    f.write("Stress-test IDs: " + str(stress_test_ids) + "\n")
```