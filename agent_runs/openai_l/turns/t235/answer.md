The check stopped at a NumPy-versus-PyTorch type mismatch; it did not test the hypothesis or change weights. I’ll fix the conversion and repeat it. Matching reconstructed logits and aligned dose contrasts are still required before fitting.

```python
from scipy.special import logsumexp, ndtr

def np212(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

print("CACHE CHECKS:")
print("W0 versus original:", np.max(np.abs(Wbase204 - np212(load_original("attr_heads.value_bin.0.weight")))))
print("W2 versus original:", np.max(np.abs(Wfinal204 - np212(load_original("attr_heads.value_bin.2.weight")))))
print("Last raw-input coordinate:", np.min(Xraw204[..., -1]), np.max(Xraw204[..., -1]))
print("Hidden reconstruction RMS:", np.sqrt(np.mean((Hcheck204.astype(float) - H179)**2)))
recon212 = Hcheck204 @ Wfinal204.T + bfinal204 + offset204
print("Reconstructed versus L179 RMS:", np.sqrt(np.mean((recon212 - L179)**2)))
print("Reconstructed versus Lorig179 RMS:", np.sqrt(np.mean((recon212 - Lorig179)**2)))
print("Fit/validation rows:", fitmask179.sum(), valmask179.sum())

print("\nPAIRED DOSE RANGES AND GATE:")
for drug in ["insulin", "norepinephrine"]:
    ids212, xi212, xf212, dr212 = cases207[(drug, 2.0)]
    d212 = directions207[(drug, .1)]
    print(drug, "ids", len(ids212),
          "rate quantiles", np.quantile(dr212, [0, .1, .5, .9, 1]),
          "pair alignment RMS", np.sqrt(np.mean((xf212 - Xraw204[:, ids212])**2)),
          "gate means", (xf212 @ gatecanon200).mean(),
          "dose contrast", np.mean((xi212-xf212) @ d212))

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
for section212 in notes210.split("\n## "):
    if section212.startswith(("Run 14", "Run 15", "Run 16", "Run 17", "Run 18", "Run 19",
                              "Runs 77", "Run 100")):
        print(section212[:1800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 208–212 / nonlinear repair setup\n"
            "Recovered canonical paired raw head inputs, family masks and dose directions. "
            "No additional weights changed. Next fit analytically with a few replaced GELU units, "
            "preserving original factual logits while enforcing record-derived dose contrasts.\n")
```
