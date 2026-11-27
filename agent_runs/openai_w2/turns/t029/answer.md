The reconstruction now has negligible channel bias, with errors consistent with the tool’s four-draw sampling noise. I’ll make a temporary dose-input probe to identify how the state responds to dose changes while the recorded readings remain fixed. That response may reveal a direction we can strengthen without materially changing factual predictions.

```python
import numpy as np
import pandas as pd
from pathlib import Path

assert edited() == [], "Unexpected existing edits."
S_reference = S
sample_stays = [int(s) for s in pd.unique(S["stay_id"])]
assert len(sample_stays) <= 100

head_original_features = local_value_head(
    head_data, reconstruction_cfg, bin_side=reconstruction_side,
    zmode="prior_mc", nsamples=64, return_features=True, seed=917
)

head_previous_hour = np.floor(head_data.hour.to_numpy() + 1e-6).astype(int) - 1
head_panel_index = pd.MultiIndex.from_arrays(
    [head_data.stay_id.to_numpy(), head_previous_hour],
    names=panel.index.names
)
head_exposures = panel.reindex(head_panel_index)[
    ["insulin", "norepinephrine"]
].fillna(0).reset_index(drop=True)
head_exposures.index = head_data.index

print("=== RECONSTRUCTED FACTUAL HEAD ===")
for event in head_data.event.unique():
    m = head_data.event.eq(event).to_numpy()
    print(event, "n:", int(m.sum()),
          "mean bias:", round(float(np.mean(
              head_original_features["mean"][m]
              - head_data.predicted_mean.to_numpy()[m])), 4))

print("\n=== PRE-DOSE HISTORY FEATURES ===")
print(F.columns.tolist())

probe_numeric_original = load_original("embed.numeric.weight")
print("\nNumeric embedding shape:", probe_numeric_original.shape)
assert tuple(probe_numeric_original.shape) == (192, 6)

# Increase dose and rate robust scores by 0.5 only where those attributes exist.
# Measurement embeddings are unchanged. This is a diagnostic probe, not the repair.
probe_delta_z = 0.5
probe_numeric = probe_numeric_original.copy()
probe_numeric[:, 3] += probe_delta_z * probe_numeric_original[:, 2]
probe_numeric[:, 5] += probe_delta_z * probe_numeric_original[:, 4]
save_edit("embed.numeric.weight", probe_numeric)

probe_rate_changes = {
    drug: probe_delta_z * numeric_specs[(drug, "rate")]["iqr"]
    for drug in ["insulin", "norepinephrine"]
}
print("\n=== TEMPORARY PROBE ===")
print("Rate-equivalent increments:", probe_rate_changes)
print("Edited tensors:", edited())
print("Training stays for the next states request:", sample_stays)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 25: successful head reconstruction; diagnostic dose probe\n"
        "Head config: GELU for gap, prior and value; query type 1; status 0; "
        "last flags (0,), age column 3, value-z column 6; ages log1p(seconds)/10; "
        "bin side left. Channel biases now negligible relative to four-draw noise.\n"
        "Saved original states as S_reference and baseline features as "
        "head_original_features (64 prior draws). Temporary edit ONLY "
        "embed.numeric.weight: presence columns 3 and 5 receive 0.5 times "
        "dose/rate numeric columns 2 and 4. Rate-equivalent increments: "
        + repr(probe_rate_changes) + ". Restore this tensor after probe states.\n"
        "Goal: derive a low-factual-variance value-head correction to dose response. "
        "Probe changes numeric scores, not bins, so validation against the exact "
        "dose-edit rollout is still essential.\n"
    )
```