The provisional lag model predicts a much larger insulin effect than the edited rollout, so the repair remains too weak if those lag estimates hold up. The zero norepinephrine estimate was a label-matching error (`ne_x2` versus `norepinephrine_x2`), not evidence of no effect. I’ll fix that comparison and recover the identification and feedback diagnostics needed to test both explanations.

```python
import inspect
import numpy as np
import pandas as pd
from pathlib import Path

print("=== CORRECTED MATCHED NOREPINEPHRINE ESTIMATES ===")
matched135 = matched134.copy()
for edit, multiplier in [("ne_x2", 1.0), ("ne_half", -0.5)]:
    mask = matched135.edit.eq(edit)
    rows = matched135.loc[mask]
    i = np.array([stay_lookup134[sid] for sid in rows.stay_id])
    h = np.rint(rows.hour).astype(int).to_numpy()
    matched135.loc[mask, "provisional_record_effect"] = (
        multiplier * estimated_paths134["norepinephrine"][i, h]
    )
active135 = matched135.loc[matched135.reached.astype(bool)]
print(active135.groupby("edit")[["shift", "provisional_record_effect"]].mean().round(4))
print("Equal-stay averages:")
print(active135.groupby(["edit", "stay_id"])[["shift", "provisional_record_effect"]]
      .mean().groupby("edit").mean().round(4))

print("\n=== IDENTIFICATION FIXTURES ===")
for drug, models in dml_models.items():
    print(drug, "folds", len(models))
    for j, model in enumerate(models):
        print(" fold", j, brief132(model))
for drug, residuals in dml_residuals.items():
    idx = np.asarray(residuals["indices"])
    print(drug, "index dtype", idx.dtype, "first", idx[:5],
          "range", (idx.min(), idx.max()))
    print("residual mean/SD:",
          float(np.mean(residuals["rd"])), float(np.std(residuals["rd"])))
    if np.issubdtype(idx.dtype, np.integer):
        print("example histories:")
        print(HX.iloc[idx[:2]][[
            "hour", "policy_glucose", "policy_map",
            "last_glucose_finger", "last_map_arterial"
        ]].to_string(index=False))

if "impulse_tables_repair" in globals():
    print("impulse_tables_repair:", brief132(impulse_tables_repair))
    if isinstance(impulse_tables_repair, dict):
        for drug, table in impulse_tables_repair.items():
            if isinstance(table, pd.DataFrame):
                print(drug)
                print(table.head(12).to_string(index=False)[:1800])
            else:
                print(drug, brief132(table))

print("\n=== RETAINED FEEDBACK DIAGNOSTIC ===")
if "feedback111" in globals():
    print(brief132(feedback111))
    if isinstance(feedback111, dict):
        for k, v in list(feedback111.items())[:10]:
            if isinstance(v, (np.ndarray,)):
                print(k, "shape", v.shape, "norm", float(np.linalg.norm(v)),
                      "range", (float(np.min(v)), float(np.max(v))))
            else:
                print(k, repr(v)[:350])
    elif isinstance(feedback111, np.ndarray):
        print("shape", feedback111.shape, "norm", float(np.linalg.norm(feedback111)),
              "first", feedback111.reshape(-1)[:12])

print("\n=== AVAILABLE REPAIR FUNCTIONS ===")
function_lines135 = []
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        if any(s in name.lower() for s in [
            "channel", "capture", "posterior", "probe", "gate",
            "context", "kernel", "head_features", "replay", "transition"
        ]):
            function_lines135.append(f"{name}{inspect.signature(obj)}")
print("\n".join(sorted(function_lines135))[:2400])

print("\n=== CHANNEL / FEEDBACK VARIABLE NAMES ===")
print(sorted(
    n for n in list(globals())
    if not n.startswith("_")
    and any(s in n.lower() for s in [
        "gate", "decoder", "channel", "feedback", "latent",
        "capture", "jacobian", "sensitivity"
    ])
))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\n## Run 135\n"
        "Corrected the NE provisional-target label mismatch (the tool uses "
        "ne_x2/ne_half). Recovered innovation-model, impulse-moment and "
        "feedback fixtures for independent causal and dynamic validation. "
        "No weights changed.\n"
    )
```