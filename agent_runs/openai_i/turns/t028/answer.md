The expanded search found a clear improvement: the previous reading’s normalized value belongs in column 6, not column 2. Reconstruction NLL is now close to the reference, making this a promising layout. I’ll validate it on all captured readings and fix the baseline-summary error. Small, unbiased discrepancies across sensors would support using this reconstruction to design an edit.

```python
import numpy as np
import pandas as pd
import torch

linear_base23 = linear_named

def head_exact(rows, z_samples=16, seed=917, **kwargs):
    def corrected_linear(*args, **kw):
        x, name = args[:2]
        if name == "last_numeric":
            x = x.clone()
            x[:, 6] = x[:, 2]
            x[:, 2] = 0.0
            args = (x,) + args[1:]
        return linear_base23(*args, **kw)
    global linear_named
    previous_linear = linear_named
    linear_named = corrected_linear
    try:
        return head2(rows, z_samples=z_samples, seed=seed, **kwargs)
    finally:
        linear_named = previous_linear

observed_bins23 = np.array([
    1 + np.searchsorted(numeric_specs[(e, "value")]["fine_bin_edges"], v, side="left")
    for e, v in zip(head_rows.event, head_rows.observed)
])
print("=== FULL RECONSTRUCTION VALIDATION ===")
for draws in [0, 32]:
    mean23, probability23, hidden23 = head_exact(head_rows, z_samples=draws)
    error23 = mean23 - head_rows.predicted_mean.to_numpy()
    nll23 = -np.log(probability23[
        np.arange(len(head_rows)), observed_bins23
    ].clip(1e-30))
    report23 = head_rows[["event", "last_value", "nll"]].copy()
    report23["has_last"] = report23.last_value.notna()
    report23["error"] = error23
    report23["error_squared"] = error23 ** 2
    report23["local_nll"] = nll23
    print("draws:", draws, "RMS:", np.sqrt(np.mean(error23 ** 2)),
          "bias:", error23.mean(), "NLL:", nll23.mean(),
          "reference NLL:", head_rows.nll.mean())
    print(report23.groupby(["event", "has_last"]).agg(
        n=("error", "size"), bias=("error", "mean"),
        mse=("error_squared", "mean"), local_nll=("local_nll", "mean"),
        reference_nll=("nll", "mean")).round(5).to_string())

print("\n=== ORIGINAL READOUTS ON EXPOSED HOURS ===")
dose_original23 = pd.read_csv("/work/fwd_2/dose_edit_original.csv")
exposed23 = dose_original23[dose_original23.reached.astype(bool)]
perstay23 = exposed23.groupby(["edit", "stay_id"])["shift"].mean()
baseline_readouts23 = perstay23.groupby("edit").agg(["mean", "std", "count"])
print(baseline_readouts23.to_string())
print("Per-hour shifts:")
print(exposed23.groupby(["edit", "hour"])["shift"].mean().unstack(0).round(3).to_string())

print("\n=== RETAINED RECORD MODELS ===")
print("panel index:", panel.index.names, "columns:", list(panel.columns))
print(panel.head(3).to_string())
print("HC columns:", list(HC.columns))
print("Controlled impulse coefficients:", irf_coefficients)
for drug, residual_array in dose_residuals.items():
    r = np.asarray(residual_array)
    print(drug, "residual percentiles:", np.nanpercentile(r, [1, 10, 50, 90, 99]))

with open("/work/notes.md", "a") as f:
    f.write("\n## Register layout resolved — Python 22–23; forward 4\n"
            "- Exhaustive input search identifies previous z_value in column 6, not column 2.\n"
            "- Flags are columns 0 and 3; age is provisionally log1p(age in hours)/10.\n"
            "- Added head_exact helper correcting inputs only; no weights changed.\n")
    f.write("Full 32-draw reconstruction: RMS %.6f, bias %.6f, NLL %.6f; reference %.6f.\n"
            % (np.sqrt(np.mean(error23 ** 2)), error23.mean(),
               nll23.mean(), head_rows.nll.mean()))
    f.write("Original exposed-hour readouts:\n" + baseline_readouts23.to_string() + "\n")
```