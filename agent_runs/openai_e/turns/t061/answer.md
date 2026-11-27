The records support a substantial negative insulin effect, although the long-run target remains provisional. The current trial strengthens that response but narrowly fails factual fit. I’ll now locate the factual errors and inspect the existing decoder helper; systematic glucose bias could be corrected without undoing the drug response, whereas scattered errors would favor a more selective edit.

```python
import numpy as np
import pandas as pd
import inspect
import dis
import io
import contextlib

paired55 = factual4_original.merge(
    factual4_trial,
    on=["stay_id", "hour", "event", "observed"],
    suffixes=("_original", "_trial"),
    validate="one_to_one"
)
paired55["mean_shift"] = paired55.predicted_mean_trial - paired55.predicted_mean_original
paired55["nll_change"] = paired55.nll_trial - paired55.nll_original
print("FACTUAL TRIAL DIAGNOSTICS")
for event55, frame55 in paired55.groupby("event"):
    print(event55, {
        "n": len(frame55),
        "original_nll": float(frame55.nll_original.mean()),
        "trial_nll": float(frame55.nll_trial.mean()),
        "original_mean_bias": float((frame55.predicted_mean_original-frame55.observed).mean()),
        "trial_mean_bias": float((frame55.predicted_mean_trial-frame55.observed).mean()),
        "mean_shift": float(frame55.mean_shift.mean()),
        "shift_sd": float(frame55.mean_shift.std()),
        "shift_quantiles": frame55.mean_shift.quantile([.05, .5, .95]).round(3).to_dict(),
        "original_rmse": float(np.sqrt(np.mean((frame55.predicted_mean_original-frame55.observed)**2))),
        "trial_rmse": float(np.sqrt(np.mean((frame55.predicted_mean_trial-frame55.observed)**2))),
    })
print("NLL CHANGE BY HOUR RANGE")
paired55["period"] = pd.cut(paired55.hour, [-np.inf, 0, 6, 12, 18, np.inf])
print(paired55.groupby(["event", "period"], observed=True).agg(
    n=("nll_change", "size"), nll_change=("nll_change", "mean"),
    mean_shift=("mean_shift", "mean")
).round(5).to_string())

print("\nTRIAL PARAMETERS")
for name55 in ("trial_strength47", "trial_direction47", "trial_b_delta",
               "trial_b_weight", "slot_fits47"):
    obj55 = globals().get(name55)
    if isinstance(obj55, np.ndarray):
        print(name55, obj55.shape, "norm", float(np.linalg.norm(obj55)))
    elif isinstance(obj55, dict):
        print(name55, {k: (type(v).__name__, getattr(v, "shape", None))
                       for k, v in obj55.items()})
    else:
        print(name55, type(obj55).__name__, str(obj55)[:150])

print("\nHELPER DEPENDENCIES")
for name55 in ("head_calc", "decoder_query", "next_means45",
               "recovery_function", "dose_covariates"):
    helper55 = globals().get(name55)
    if inspect.isfunction(helper55):
        print(name55, "globals:", helper55.__code__.co_names,
              "locals:", helper55.__code__.co_varnames)

print("\nVALUE-HEAD HELPER")
try:
    print(inspect.getsource(head_calc)[:3500])
except (OSError, TypeError):
    buffer55 = io.StringIO()
    with contextlib.redirect_stdout(buffer55):
        dis.dis(head_calc)
    print(buffer55.getvalue()[:4200])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 55: factual-error localization\n"
        "Compared paired original/trial means and NLL by event and time, "
        "and recovered the value-head helper dependencies. Record evidence "
        "supports insulin beta about -2.28 and provisional persistence .857; "
        "NE beta +3.25 needs stronger identification checks. No weights "
        "changed. Python runs: 55; forwards: 5.\n"
    )
```