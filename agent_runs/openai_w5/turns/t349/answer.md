The glucose signal neurons have opposite output columns to their controls, allowing a small sensor-specific offset without changing dose encoding. I’ll try a conservative offset estimated from half the tuning stays. It remains provisional: paired loss on the other half must improve, and rollout effects must remain stable.

```python
from safetensors.torch import save_file

BASE295 = {
    n: torch.as_tensor(array293(load(n))).clone().contiguous()
    for n in edited()
}
save_file(BASE295, "/work/checkpoint_run295.safetensors")

cal295 = audit293[audit293.stay_id % 2 == 1]
val295 = audit293[audit293.stay_id % 2 == 0]
bias295 = torch.as_tensor(
    array293(load("attr_heads.value_bin.0.bias"))
).clone()
calibration295 = {}

for event in ("glucose_finger", "glucose_lab"):
    sig, ctrl = sensor_neurons[event]
    centres = np.asarray(VOCAB["value_bin_centres"][event], dtype=float)
    column = W2294[1:, sig].astype(float)
    slope, intercept = np.polyfit(centres, column, 1)
    d = cal295[cal295.event == event]
    residual = d.predicted_mean.to_numpy() - d.observed.to_numpy()
    mean_bias = float(residual.mean())
    variance_proxy = float(np.mean(residual ** 2))
    derivative_proxy = slope * variance_proxy
    assert derivative_proxy < 0 and np.isfinite(derivative_proxy)

    # Conservative moment-based calibration, not an optimizer over model weights.
    delta = float(np.clip(-0.8 * mean_bias / derivative_proxy, -0.05, 0.05))
    bias295[sig] += delta
    calibration295[event] = {
        "signal_neuron": int(sig),
        "delta_bias": delta,
        "calibration_mean_bias": mean_bias,
        "validation_mean_bias_before": float(
            val295.loc[val295.event == event, "bias_edited"].mean()),
        "approximate_mean_shift": derivative_proxy * delta,
        "output_slope": float(slope),
        "slope_fit_rmse": float(np.sqrt(np.mean(
            (column - (slope * centres + intercept)) ** 2)))
    }

save_edit("attr_heads.value_bin.0.bias", bias295.contiguous())
print(json.dumps(calibration295, indent=2))
print("Only first-layer glucose signal biases changed; "
      "dose coefficients, persistence, output strengths and NE unchanged.")

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 295: reversible glucose-intercept calibration trial\n"
            "- Saved pre-trial weights in checkpoint_run295.safetensors.\n"
            "- Used odd-ID tuning stays for conservative moment-based glucose "
            "offsets; even-ID stays are the calibration validation subset.\n"
            f"- Trial offsets: {json.dumps(calibration295)}\n"
            "- No gradients or weight optimizer used. Drug effect estimates "
            "remain those from training records, not these calibration residuals.\n"
            "- This candidate is unaccepted pending paired factual NLL and "
            "finite-rollout checks; tau=1250 and all drug coefficients unchanged.\n")
```