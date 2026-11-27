The latent weights are not zero, yet changing the existing latent coordinates barely moves predictions. The dose residuals are concentrated in active-drug bundles, which is encouraging, but we still need a usable state route. I’ll perturb the latent token directly with finite differences large enough to avoid numerical rounding. This will confirm or reject whether a modest token change can control glucose and MAP separately.

```python
import time

print("Codes:", VOCAB["codes"])
print("Posterior mean range:",
      float(post93.min()), float(post93.max()),
      "RMS", float(torch.sqrt((post93 ** 2).mean())))
ztoken110 = linear63(post93, "z_in")
print("Latent token RMS:", float(torch.sqrt((ztoken110 ** 2).mean())))
print("Probe sensor counts:",
      np.unique(np.asarray(row_events94)[rows93], return_counts=True))

original_bias110 = w63("z_in.bias").clone()

def token_probe110(addition):
    previous = w63_cache["z_in.bias"]
    try:
        w63_cache["z_in.bias"] = (
            original_bias110 +
            torch.as_tensor(addition, dtype=torch.float32)
        )
        return latent_probe93(np.zeros((len(ids93), 32), dtype=np.float32))
    finally:
        w63_cache["z_in.bias"] = previous

baseline110 = token_probe110(np.zeros(192, dtype=np.float32))
print("Token probe baseline agreement:",
      float(np.sqrt(np.mean((baseline110 - base93) ** 2))))

rng110 = np.random.default_rng(110)
basis110 = np.linalg.qr(rng110.normal(size=(192, 48)))[0].astype(np.float32)
Gtoken110 = np.zeros((len(rows93), basis110.shape[1]), dtype=np.float64)
start110 = time.time()
for j110 in range(basis110.shape[1]):
    plus110 = token_probe110(2.0 * basis110[:, j110])
    minus110 = token_probe110(-2.0 * basis110[:, j110])
    Gtoken110[:, j110] = (plus110 - minus110) / 4.0
print("Elapsed seconds:", round(time.time() - start110, 2))
print("Token sensitivity RMS/max:",
      float(np.sqrt(np.mean(Gtoken110 ** 2))),
      float(np.max(np.abs(Gtoken110))))

sensor_codes110 = np.asarray(row_events94)[rows93]
for sensor110 in np.unique(sensor_codes110):
    m110 = sensor_codes110 == sensor110
    print("sensor", sensor110, "n", int(m110.sum()),
          "mean sensitivity norm",
          float(np.linalg.norm(Gtoken110[m110].mean(axis=0))),
          "row sensitivity RMS",
          float(np.sqrt(np.mean(Gtoken110[m110] ** 2))))

print("Singular values:", np.round(np.linalg.svd(Gtoken110, compute_uv=False)[:12], 6))
print("Edits remain:", edited())

with open("/work/notes.md", "a") as f110:
    f110.write(
        "\n## Run 110 / direct latent-token controllability\n"
        "Testing 48 orthonormal token directions with central finite differences "
        "(±2), using validated recurrence and decoder. No weights saved. This "
        "separates a weak existing z-coordinate route from weak token controllability.\n"
    )
```
