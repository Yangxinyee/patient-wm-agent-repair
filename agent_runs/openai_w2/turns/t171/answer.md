The candidate matrices are available, and the records still support effects larger than the original rollouts predict, though accumulation remains uncertain. I’ll test the initial-safe decoder at gains −10 and +12—the gains that previously exceeded the factual budget. If removing the initial signal fixes that failure, the paired factual test should show unchanged early readings and an NLL increase below 0.005.

```python
import pandas as pd
import torch

def as_np150(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().copy()
    return np.asarray(x).copy()

candidate_key150 = (0.3, 0.0001)
candidate_W150 = constraint_candidates147[candidate_key150].copy()
candidate_gains150 = np.array([-10.0, 12.0])
original_B150 = as_np150(load_original("core.b_in.weight"))
delta_scale150 = np.logaddexp(
    0.0, as_np150(load_original("core.delta_in")).reshape(-1)
)
assert original_B150.shape == (192, 192)
assert delta_scale150.shape == (192,)
assert np.all(delta_scale150 > 0)

reconstructed_fallback150 = (
    original_B150
    + ((V128 * np.array([-5.0, 8.0])[None, :]) @ Wdose134.T)
      / delta_scale150[:, None]
).astype(np.float32)
current_B150 = as_np150(load("core.b_in.weight"))
print("Fallback reconstruction max error:",
      np.max(np.abs(current_B150 - reconstructed_fallback150)))
assert np.allclose(current_B150, fallback_recurrent139, atol=2e-5, rtol=2e-6)
assert np.allclose(current_B150, reconstructed_fallback150, atol=2e-5, rtol=2e-6)
initial_safe_fallback150 = current_B150.copy()

print("Initial-safe decoder diagnostics:")
print(report_decoder147(candidate_W150))

candidate_B150 = (
    original_B150
    + ((V128 * candidate_gains150[None, :]) @ candidate_W150.T)
      / delta_scale150[:, None]
).astype(np.float32)
assert np.isfinite(candidate_B150).all()
save_edit("core.b_in.weight", torch.from_numpy(candidate_B150))
print("Installed gains:", candidate_gains150.tolist())
print("Edited tensors:", edited())

factual_reference150 = pd.read_csv("/work/fwd_13/factual_original.csv")
print("Factual columns:", factual_reference150.columns.tolist())
cohort150 = factual_reference150["stay_id"].drop_duplicates().tolist()
print("Paired factual cohort:", cohort150)
print("Cohort size:", len(cohort150))

with open("/work/notes.md", "a") as f150:
    f150.write(
        "\n## Runs 148–150: initial-safe stronger candidate\n"
        "Recovered the architecture, effect evidence, and constrained decoders. "
        "Original fallback [-5,+8] was verified against its canonical construction "
        "and saved as initial_safe_fallback150. Installed only core.b_in.weight "
        "using constraint_candidates147[(0.3,0.0001)] and gains [-10,+12]. "
        "This decoder constrains the initial four slot features to zero dose signal. "
        "Require paired factual NLL increase <=0.005 and no initial-reading distortion "
        "before rollout. Record-supported accumulation remains uncertain; "
        "the previous validated fallback remains available.\n"
    )
```
