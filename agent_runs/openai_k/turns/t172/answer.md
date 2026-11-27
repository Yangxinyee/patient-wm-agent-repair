The records support substantial, drug-specific effects across both glucose sensors and both MAP sensors. The contrast alignment failed because a bundle can contain both drugs, so a bundle index alone is not a unique key. I’ll align by drug and bundle, then compute the sensitivity matrix; the recovered contrasts must cover every sampled context without ambiguity.

```python
import time, numpy as np, torch

print("=== DYNAMIC IDENTIFICATION AND CHECKS ===")
print(open("/work/notes.md").read()[3500:6500])

# A bundle may occur twice: once for each drug contrast.
ucf156 = context151["u"].clone()
dc156 = np.zeros(len(indices151), dtype=np.float64)
drug156 = np.asarray(drug151, dtype=int).copy()
for c157 in CF89:
    d157 = int(c157["drug"])
    lookup157 = {int(v): j for j, v in enumerate(c157["indices"])}
    p157 = np.flatnonzero(drug156 == d157)
    q157 = np.array([lookup157[int(indices151[i])] for i in p157])
    ucf156[p157] = c157["u"][q157]
    dc156[p157] = np.asarray(c157["dc"])[q157]
assert np.all(np.isfinite(dc156)) and np.all(dc156 > 0)
print("aligned contexts:", len(dc156),
      "unique bundles:", len(np.unique(indices151)),
      "drug counts:", np.bincount(drug156).tolist())

zcf156, tokencf156 = posterior_token154(zero_weights154, ucf156)
ocf156, hcf156 = local_token154(tokencf156, ucf156)
baseline_contrast156 = dynamic_delta154(
    dynamic_views154(ocf156, hcf156), views_base154
)
dz156 = (zcf156[:, 30:32] - z_base154[:, 30:32]).numpy()

print("\n=== EXISTING RESPONSE PER PHYSICAL RATE UNIT ===")
print("event order:", events151)
for d157 in range(2):
    q157 = drug156 == d157
    print("drug", d157,
          "baseline response",
          np.round((baseline_contrast156[q157]/dc156[q157, None]).mean(0), 4),
          "channel contrast/rate",
          np.round((dz156[q157]/dc156[q157, None]).mean(0), 4),
          "factual channel SD",
          np.round(z_base154[q157, 30:32].numpy().std(0), 4))

J156 = np.empty((len(indices151), len(events151), 192), dtype=np.float32)
eps156 = 0.04
start156 = time.perf_counter()
for j157 in range(192):
    perturb157 = torch.zeros(192, dtype=torch.float32)
    perturb157[j157] = eps156
    op157, hp157 = local_token154(token_base154 + perturb157[None, :])
    om157, hm157 = local_token154(token_base154 - perturb157[None, :])
    J156[:, :, j157] = dynamic_delta154(
        dynamic_views154(op157, hp157),
        dynamic_views154(om157, hm157)
    ) / (2*eps156)
    if (j157+1) % 64 == 0:
        print("Jacobian coordinates", j157+1,
              "elapsed seconds", round(time.perf_counter()-start156, 1))

print("\n=== JACOBIAN CHECK ===")
for d157 in range(2):
    check157 = np.einsum("nek,k->ne", J156, D89[:, d157])
    print("direction", d157, "RMS discrepancy",
          round(float(np.sqrt(np.mean(
              (check157-responses154[d157])**2
          ))), 6))
print("finite:", bool(np.isfinite(J156).all()),
      "event RMS sensitivities:",
      np.round(np.sqrt(np.mean(J156**2, axis=(0, 2))), 4))

with open("/work/notes.md", "a") as f157:
    f157.write(
        "\n## Runs 156-157: drug/bundle alignment and exact-core Jacobian\n"
        "- Run 156 stopped before differentiation because joint-drug bundles require drug-specific alignment keys.\n"
        "- Corrected alignment by drug and bundle; computed central finite-difference token/readout sensitivities.\n"
        "- Record cross-fitted immediate effects: finger -1.935, lab -2.015 per insulin unit; arterial +3.053, cuff +2.528 per NE unit.\n"
        "- No saved weights changed. Python runs: 157; forward passes: 10.\n"
    )
```