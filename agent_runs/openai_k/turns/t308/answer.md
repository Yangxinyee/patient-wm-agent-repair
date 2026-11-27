The audit confirmed how the recurrent edit enters the model. The current glucose edit passed the first factual batch, but the MAP correction remains unvalidated. I’ll compare the lower-distortion MAP routes locally; stronger dose responses with less factual disturbance would support testing one officially.

```python
import numpy as np, torch, inspect
torch.set_num_threads(8)

def r272_summary(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return {"shape": x.shape,
                "mean": float(np.mean(x)),
                "sd": float(np.std(x))}
    if isinstance(x, dict):
        return {str(k): r272_summary(v) for k, v in list(x.items())[:12]}
    if isinstance(x, (list, tuple)):
        return {"length": len(x),
                "items": [r272_summary(v) for v in x[:5]]}
    if callable(x):
        return str(inspect.signature(x))
    return str(x)[:250]

print("=== RETAINED FISHER FITS ===")
for name, value in list(globals().items()):
    if name.startswith(("r267_", "r268_")):
        print(name, r272_summary(value))

r272_dicts = [
    (name, value) for name, value in list(globals().items())
    if isinstance(value, dict)
    and ("factual", 1.0) in value
    and isinstance(value[("factual", 1.0)], dict)
    and "pre" in value[("factual", 1.0)]
    and "E" in value[("factual", 1.0)]
]
assert r272_dicts, "No retained replay cases found."
r272_case_name, r272_cases = max(r272_dicts, key=lambda item: len(item[1]))
print("Replay cases:", r272_case_name, list(r272_cases))
print("Readout indices:", r272_summary(indices151))

r272_delta = np242(r245_delta).astype(np.float64)
r272_v = np242(rr_Nv).astype(np.float64)
r272_old_map = np.outer(np242(rr_Ndirection) / r272_delta, r272_v)
print("MAP factorization relative error:",
      float(np.linalg.norm(r272_old_map - np242(rr_Ntotal)) /
            np.linalg.norm(np242(rr_Ntotal))))

r272_glucose = 1.15 * np242(r255_CI)
r272_base = {
    key: r256_predict(r272_cases[key], r272_glucose)
    for key in [("factual", 1.0),
                ("norepinephrine", 2.0),
                ("norepinephrine", 0.5)]
}
print("Full-distribution helper:",
      r272_summary(r267_full_pre(
          r272_cases[("factual", 1.0)]["pre"], indices151)))

print("=== LOCAL MAP ROUTE COMPARISON ===")
r272_trials = {}
for ridge, direction in r267_routes.items():
    correction = np.outer(np242(direction) / r272_delta, r272_v)
    predictions = {
        key: r256_predict(r272_cases[key], r272_glucose + correction)
        for key in r272_base
    }
    factual_delta = predictions[("factual", 1.0)] - r272_base[("factual", 1.0)]
    result = {
        "factual_mean_shift": np.mean(factual_delta, axis=0).round(5).tolist(),
        "factual_RMS_shift": np.sqrt(np.mean(factual_delta**2, axis=0)).round(5).tolist()
    }
    for multiplier in [2.0, 0.5]:
        key = ("norepinephrine", multiplier)
        result[f"NE_{multiplier}_dose_shift"] = np.mean(
            predictions[key] - predictions[("factual", 1.0)], axis=0
        ).round(5).tolist()
        result[f"NE_{multiplier}_added_response"] = np.mean(
            (predictions[key] - predictions[("factual", 1.0)])
            - (r272_base[key] - r272_base[("factual", 1.0)]), axis=0
        ).round(5).tolist()
    r272_trials[ridge] = {"correction": correction, "summary": result}
    print("ridge", ridge, result)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 272: compare minimum-distortion MAP routes\n"
        "- Recovered retained Fisher-route fits and checked the recurrent injection factorization.\n"
        "- Compared local factual mean disturbance and NE doubled/halved responses with the same dose feature.\n"
        "- No weights changed; local results are not official factual or rollout validation.\n"
        "- Python runs: 272; forward passes: 24.\n"
    )
```