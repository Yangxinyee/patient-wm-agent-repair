The paired inputs and sensor-family gate are usable. I’ll now check their alignment, the original logit reconstruction, and the correction targets before fitting a replacement head. Exact reconstruction and correctly signed dose contrasts would confirm that the experiment is testing the intended mechanism.

```python
from scipy.special import logsumexp, ndtr

print("CACHE CHECKS:")
print("W0 versus original:", np.max(np.abs(Wbase204 - load_original("attr_heads.value_bin.0.weight"))))
print("W2 versus original:", np.max(np.abs(Wfinal204 - load_original("attr_heads.value_bin.2.weight"))))
print("Last raw-input coordinate:", np.min(Xraw204[..., -1]), np.max(Xraw204[..., -1]))
print("Hidden reconstruction RMS:", np.sqrt(np.mean((Hcheck204.astype(float) - H179)**2)))
recon211 = Hcheck204 @ Wfinal204.T + bfinal204 + offset204
print("Reconstructed versus L179 RMS:", np.sqrt(np.mean((recon211 - L179)**2)))
print("Reconstructed versus Lorig179 RMS:", np.sqrt(np.mean((recon211 - Lorig179)**2)))
print("Fit/validation rows:", fitmask179.sum(), valmask179.sum())

print("\nPAIRED DOSE RANGES AND GATE:")
for drug in ["insulin", "norepinephrine"]:
    ids211, xi211, xf211, dr211 = cases207[(drug, 2.0)]
    d211 = directions207[(drug, .1)]
    print(drug, "ids", len(ids211),
          "rate quantiles", np.quantile(dr211, [0, .1, .5, .9, 1]),
          "pair alignment RMS", np.sqrt(np.mean((xf211 - Xraw204[:, ids211])**2)),
          "gate means", (xf211 @ gatecanon200).mean(),
          "dose contrast", np.mean((xi211-xf211) @ d211))

print("\nTARGET STRUCTURE:", describe210(targets179))
print("Other target structures:", describe210(mean_tilt179), describe210(beta179))
print("Value NLL helper:")
try:
    print(inspect.getsource(nll179))
except Exception:
    print("code names", nll179.__code__.co_names)
    print("constants", nll179.__code__.co_consts)
print("Original cached NLL:", original_nll179, nll179(Lorig179))

print("\nCAUSAL NOTES (accepted early evidence):")
for section211 in notes210.split("\n## "):
    if section211.startswith(("Run 14", "Run 15", "Run 16", "Run 17", "Run 18", "Run 19",
                              "Runs 77", "Run 100")):
        print(section211[:1800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 208–211 / nonlinear repair setup\n"
            "Recovered canonical paired raw head inputs, family masks and dose directions. "
            "No additional weights changed. Next fit analytically with a few replaced GELU units, "
            "preserving original factual logits while enforcing record-derived dose contrasts.\n")
```
