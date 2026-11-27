The failed request supplied no evidence and used no forward pass. I won’t guess another ID range. I’ll use the last Python run to recover valid IDs, verify the installed trial, and update the notebook. After this audit, the weights will be frozen; the remaining forward passes will test whether the trial actually satisfies the factual guardrail.

```python
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

# Read the architecture before inspecting the model or records.
architecture = Path("/model/ARCHITECTURE.md").read_text()
print("Architecture read:", len(architecture), "characters")

def stay_column(columns):
    columns = list(columns)
    for name in ("stay_id", "stay", "icu_stay_id", "episode_id", "stayid"):
        if name in columns:
            return name
    candidates = [c for c in columns if "stay" in c.lower()]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("Cannot identify stay column: " + repr(columns))

def canonical_id(x):
    s = str(x).strip()
    if re.fullmatch(r"-?(0|[1-9]\d*)", s):
        return int(s)
    return s

# Recover IDs from records, corroborated by successful factual requests.
header = pd.read_csv("/data/tuning.csv", nrows=0)
sid = stay_column(header.columns)
raw_ids = pd.read_csv(
    "/data/tuning.csv", usecols=[sid], dtype={sid: "string"}
)[sid].dropna().unique()
tuning_ids = [canonical_id(x) for x in raw_ids]
tuning_lookup = {str(x): x for x in tuning_ids}
successful_ids = set()

def forward_number(path):
    return int(path.parent.name.split("_")[-1])

original_files = sorted(
    Path("/work").glob("fwd_*/factual_original.csv"),
    key=forward_number
)
for path in original_files:
    try:
        columns = pd.read_csv(path, nrows=0).columns
        c = stay_column(columns)
        observed = pd.read_csv(path, usecols=[c], dtype={c: "string"})[c]
        successful_ids.update(
            str(canonical_id(x)) for x in observed.dropna().unique()
            if str(canonical_id(x)) in tuning_lookup
        )
    except Exception as exc:
        print("ID recovery warning:", str(path), repr(exc))

eligible_ids = [x for x in tuning_ids if str(x) in successful_ids]
if not eligible_ids:
    raise RuntimeError("No tuning IDs corroborated by successful factual requests.")
eligible_ids = sorted(eligible_ids, key=lambda x: (isinstance(x, str), x))
batches = [eligible_ids[i:i+100] for i in range(0, len(eligible_ids), 100)]
print("\n=== VERIFIED TUNING IDS ===")
print("Tuning stays:", len(tuning_ids),
      "previously successful tuning stays:", len(eligible_ids))
for i, batch in enumerate(batches, 1):
    print("BATCH", i, json.dumps(batch))

# Verify exact provenance. Roll back only if the installed trial is corrupted
# or differs from its baseline in an unexpected tensor.
torch.set_num_threads(8)
live_names = set(edited())
allowed_changes = {
    "core.flow.a",
    "attr_heads.value_bin.0.weight",
    "attr_heads.value_bin.0.bias",
}
trial_mismatches = []
baseline_changes = []
baseline_path = "/work/checkpoint_baseline298.safetensors"
trial_path = "/work/checkpoint_trial299.safetensors"

def cpu_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device="cpu")

with safe_open(baseline_path, framework="pt", device="cpu") as base, \
     safe_open(trial_path, framework="pt", device="cpu") as trial:
    base_keys, trial_keys = set(base.keys()), set(trial.keys())
    names = sorted((live_names | base_keys | trial_keys) & set(NAMES))
    for name in names:
        current = cpu_tensor(load(name))
        expected = (trial.get_tensor(name) if name in trial_keys
                    else cpu_tensor(load_original(name)))
        baseline = (base.get_tensor(name) if name in base_keys
                    else cpu_tensor(load_original(name)))
        trial_difference = float((current - expected).abs().max())
        base_difference = float((current - baseline).abs().max())
        if trial_difference > 1e-6:
            trial_mismatches.append((name, trial_difference))
        if base_difference > 1e-6:
            baseline_changes.append((name, base_difference))

unexpected = [x for x in baseline_changes if x[0] not in allowed_changes]
restored = bool(trial_mismatches or unexpected)
if restored:
    with safe_open(baseline_path, framework="pt", device="cpu") as base:
        base_keys = set(base.keys()) & set(NAMES)
        for name in list(edited()):
            if name not in base_keys:
                drop_edit(name)
        for name in sorted(base_keys):
            save_edit(name, base.get_tensor(name))
    retained = "exact 1250-second baseline (provenance audit rollback)"
else:
    retained = "1100-second insulin persistence/centering trial"

print("\n=== FINAL WEIGHT AUDIT ===")
print("Trial mismatches:", trial_mismatches)
print("Changes from baseline:", baseline_changes)
print("Unexpected changes:", unexpected)
print("Retained:", retained)
print("Edited tensor count:", len(edited()))

# Recover paired historical factual summaries without mixing checkpoints.
print("\n=== RECENT HISTORICAL FACTUAL FILES ===")
for original_path in original_files[-12:]:
    edited_path = original_path.with_name("factual_edited.csv")
    if not edited_path.exists():
        continue
    try:
        original = pd.read_csv(original_path)
        changed = pd.read_csv(edited_path)
        nll_columns = [
            c for c in original.columns
            if "nll" in c.lower() and c in changed.columns
            and pd.api.types.is_numeric_dtype(original[c])
            and pd.api.types.is_numeric_dtype(changed[c])
        ]
        summary = {
            "forward": forward_number(original_path),
            "original_rows": len(original),
            "edited_rows": len(changed),
            "stays": int(original[stay_column(original.columns)].nunique()),
            "mean_nll": {
                c: {
                    "original": float(original[c].mean()),
                    "edited": float(changed[c].mean()),
                    "difference": float(changed[c].mean() - original[c].mean())
                } for c in nll_columns
            }
        }
        print(json.dumps(summary))
    except Exception as exc:
        print("Historical-summary warning:", repr(exc))

# Recover the latest rollout schema/results for interpreting new requests.
dose_files = sorted(
    Path("/work").glob("fwd_*/dose_edit_edited.csv"),
    key=forward_number
)
print("\n=== LATEST HISTORICAL ROLLOUTS ===")
for path in dose_files[-2:]:
    try:
        frame = pd.read_csv(path)
        print(str(path), "columns:", list(frame.columns))
        label = next(
            (c for c in ("edit", "intervention", "dose_edit", "scenario")
             if c in frame.columns), None
        )
        shift_columns = [
            c for c in frame.columns
            if any(word in c.lower() for word in ("shift", "delta", "change"))
            and pd.api.types.is_numeric_dtype(frame[c])
        ]
        if label and shift_columns:
            print(frame.groupby(label)[shift_columns].mean().to_string())
        else:
            print(frame.head(8).to_string(index=False))
    except Exception as exc:
        print("Rollout-summary warning:", repr(exc))

notes_path = Path("/work/notes.md")
notes = notes_path.read_text()
print("\n=== EARLY RECORD-ANALYSIS NOTES ===")
print(notes[:6000])
entry = f"""
## Run 300: final provenance audit and exact ID recovery
- The attempted numeric tuning-ID range was invalid; that request was not counted
  and supplied no evidence about the model.
- Recovered {len(eligible_ids)} tuning IDs corroborated by successful factual files,
  partitioned into {len(batches)} batches of at most 100.
- Retained candidate: {retained}.
- Trial mismatches: {trial_mismatches}.
- Changes from baseline: {baseline_changes}.
- Unexpected changes: {unexpected}.
- No gradient training or weight optimization was performed.
- Python budget is exhausted: 300/300. Forward budget remains 42/50.
- Weights are now frozen. The remaining forward passes must audit the retained
  candidate on paired factual predictions and finite dose rollouts.
- For the 1100-second trial, the prior 400-stay baseline delta of +0.00414485
  is NOT its measured loss. Full factual acceptance remains unverified.
- Record-derived drug-effect estimates remain assumption-dependent, not observed
  intervention truth. Total non-value loss and unsupported interventions remain
  unverified.
"""
notes_path.write_text(notes + entry)
print("\nNotebook updated. No further Python runs remain.")
```