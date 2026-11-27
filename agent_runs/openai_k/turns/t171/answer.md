The exact update reproduces the original state, but the old directions have variable sensitivity and some cross-drug effects. That makes further scaling of those directions unreliable. I’ll measure the full token-to-reading response and separate the model’s existing dose response from the correction needed; a held-out, drug-specific fit would justify testing new directions officially.

```python
import time, numpy as np, torch

print("=== EARLY IDENTIFICATION NOTES ===")
print(open("/work/notes.md").read()[:3500])

# Align the retained doubled-dose contrasts with the captured contexts.
position156 = {int(v): i for i, v in enumerate(indices151)}
ucf156 = context151["u"].clone()
dc156 = np.zeros(len(indices151), dtype=np.float64)
drug156 = np.full(len(indices151), -1, dtype=int)
for c156 in CF89:
    p156 = np.array([position156[int(v)] for v in c156["indices"]])
    ucf156[p156] = c156["u"]
    dc156[p156] = np.asarray(c156["dc"])
    drug156[p156] = int(c156["drug"])
assert np.all(drug156 >= 0) and np.all(dc156 > 0)

zcf156, tokencf156 = posterior_token154(zero_weights154, ucf156)
ocf156, hcf156 = local_token154(tokencf156, ucf156)
baseline_contrast156 = dynamic_delta154(
    dynamic_views154(ocf156, hcf156), views_base154
)
dz156 = (zcf156[:, 30:32] - z_base154[:, 30:32]).numpy()

print("\n=== EXISTING RESPONSE PER PHYSICAL RATE UNIT ===")
print("event order:", events151)
for d156 in range(2):
    q156 = drug156 == d156
    print("drug", d156,
          "baseline response",
          np.round((baseline_contrast156[q156]/dc156[q156, None]).mean(0), 4),
          "channel contrast/rate",
          np.round((dz156[q156]/dc156[q156, None]).mean(0), 4),
          "factual channel SD",
          np.round(z_base154[q156, 30:32].numpy().std(0), 4))

# Numerical differentiation only: no gradients, backpropagation, or optimizer.
J156 = np.empty((len(indices151), len(events151), 192), dtype=np.float32)
eps156 = 0.04
start156 = time.perf_counter()
for j156 in range(192):
    perturb156 = torch.zeros(192, dtype=torch.float32)
    perturb156[j156] = eps156
    op156, hp156 = local_token154(token_base154 + perturb156[None, :])
    om156, hm156 = local_token154(token_base154 - perturb156[None, :])
    J156[:, :, j156] = dynamic_delta154(
        dynamic_views154(op156, hp156),
        dynamic_views154(om156, hm156)
    ) / (2*eps156)
    if (j156+1) % 64 == 0:
        print("Jacobian coordinates", j156+1,
              "elapsed seconds", round(time.perf_counter()-start156, 1))

print("\n=== JACOBIAN CHECK AGAINST EARLIER ASSAY ===")
for d156 in range(2):
    check156 = np.einsum("nek,k->ne", J156, D89[:, d156])
    print("direction", d156, "RMS discrepancy",
          round(float(np.sqrt(np.mean(
              (check156-responses154[d156])**2
          ))), 6))
print("Jacobian finite:", bool(np.isfinite(J156).all()),
      "event RMS sensitivities:",
      np.round(np.sqrt(np.mean(J156**2, axis=(0, 2))), 4))

with open("/work/notes.md", "a") as f156:
    f156.write(
        "\n## Run 156: full exact-core token/readout Jacobian\n"
        "- Aligned doubled physical-dose contrasts and separated their existing model response from reserved-channel contrasts.\n"
        "- Computed a 192-coordinate central finite-difference Jacobian with fixed strictly prior reading registers.\n"
        "- Numerical differentiation is not gradient training; no saved weights changed.\n"
        "- Python runs: 156; forward passes: 10.\n"
    )
```