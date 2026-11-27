The reconstruction resolved the problem: the cached raw and folded decoder names were swapped. The corrected decoder matches the regression, with factual residual RMS of 0.107 for insulin and 0.033 for norepinephrine. I’ll reinstall the conservative candidate using that decoder, then test factual NLL; only a passing result will justify a rollout test.

```python
assert not edited()
assert np.max(np.abs(pred134 - Efull134 @ Wdose134)) < 1e-3

candidate_gain135 = np.array([-5.0, 8.0])
change_b135 = ((V128 * candidate_gain135[None, :]) / delta132[:, None]) @ Wdose134.T
candidate_b135 = (original_b132 + change_b135).astype(np.float32)

actual_injection135 = (
    Efull134 @ (candidate_b135.astype(np.float64) - original_b132).T
) * delta132
expected_injection135 = (pred134 * candidate_gain135) @ V128.T
injection_error135 = np.max(np.abs(actual_injection135 - expected_injection135))
print("Float32 implementation max error:", injection_error135)
print("Factual scalar displacement RMS:",
      np.sqrt(np.mean((pred134 * candidate_gain135)**2, axis=(0, 1))))
print("Original/change weight norms:",
      np.linalg.norm(original_b132), np.linalg.norm(change_b135))
assert injection_error135 < 0.005

save_edit("core.b_in.weight", torch.from_numpy(candidate_b135))

with open("/work/notes.md", "a") as f:
    f.write("""
## Run 135: corrected recurrent candidate installed
Resolved the cached-name swap: Wdose128 was RAW slopes, whereas Edecoder128 was already FOLDED. Canonical construction is raw_coef134=B[1:]/sd, affine_b134=B[0]-mu@raw_coef134, Wdose134=raw_coef134+outer(1/gamma,affine_b134/sum(beta/gamma)).
Matches standardized factual predictions exactly and folded predictions within 2.5e-5. Factual residual RMS 0.10747 insulin, 0.03304 NE.
Installed only core.b_in.weight using gains [-5,+8] and V128/softplus(delta_in). Large matrix norm reflects the intercept folded through the layer-normalization invariant; actual factual injection is small. Await exact paired factual test, then rollout if budget holds. Run-132 construction must not be reused.
""")
print("Current edits:", edited())
```