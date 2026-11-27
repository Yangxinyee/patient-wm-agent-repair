The refit reduced latent noise by orders of magnitude, and the stricter gate nearly separates valid histories from the rest. I’ll test the graft’s paired factual cost, then provisionally install the −2 candidate only if that check is comfortably below the limit. GPU factual scoring will determine whether it is safe.

```python
# Refit the selected small regression using the two training-state batches.
projector3 = robust_projectors["register_prior"]
xpool3 = np.concatenate([
    robust_X[s3][:, np.asarray(valid_glucose_masks[s3], bool)].reshape(-1, 192)
    for s3 in ["first", "extra"]
]) @ projector3
ypool3 = np.concatenate([
    np.tile(np.asarray(targets0[s3])[np.asarray(valid_glucose_masks[s3], bool)],
            robust_samples)
    for s3 in ["first", "extra"]
])
loc3 = xpool3.mean(0)
scale3 = np.maximum(xpool3.std(0), 0.05)
reg3 = Ridge(alpha=0.1).fit((xpool3-loc3)/scale3, ypool3)
graft_decoder_w = (projector3 @ (reg3.coef_/scale3)).astype(np.float32)
graft_decoder_b = float(reg3.intercept_ - loc3 @ (reg3.coef_/scale3))

gatepool3 = np.concatenate([robust_X[s3].reshape(-1, 192)
                            for s3 in ["first", "extra"]])
gateypool3 = np.concatenate([
    np.tile(np.where(np.asarray(valid_glucose_masks[s3], bool), 1.0, -1.0),
            robust_samples)
    for s3 in ["first", "extra"]
])
gateloc3 = gatepool3.mean(0)
gatescale3 = np.maximum(gatepool3.std(0), 0.05)
gatereg3 = Ridge(alpha=10.0).fit(
    (gatepool3-gateloc3)/gatescale3, gateypool3)
gate_w = (gatereg3.coef_/gatescale3).astype(np.float32)
gate_b = float(gatereg3.intercept_ - gateloc3 @ gate_w - 0.55)
graft_gate_scale = 20.0

orig_t3 = {n3: torch.from_numpy(np.asarray(a3).copy()).float()
           for n3, a3 in graft_original.items()}
trans3 = robust_parts["tuning"]["transition"]
xdraws3 = [torch.from_numpy(a3) for a3 in robust_X["tuning"]]
pbase3 = sum(graft_probs(a3, trans3, orig_t3) for a3 in xdraws3)/robust_samples
bins3 = head_obsbins.long()
idx3 = torch.arange(len(bins3))
nllbase3 = -pbase3[idx3, bins3].clamp_min(1e-20).log()
events3 = decode_meta_sets["tuning"]["event"].to_numpy()
centres3 = torch.as_tensor(np.asarray(cm), dtype=torch.float32)

robust_candidates3 = {}
robust_costs3 = {}
print("PAIRED CPU FACTUAL CHECK")
for beta3 in [0.0, -2.0, -4.0, -8.0]:
    candidate3 = make_insulin_graft(beta3)
    tensors3 = {n3: torch.from_numpy(a3.copy()).float()
                for n3, a3 in candidate3.items()}
    pp3 = sum(graft_probs(a3, trans3, tensors3)
              for a3 in xdraws3)/robust_samples
    nn3 = -pp3[idx3, bins3].clamp_min(1e-20).log()
    shifts3 = ((pp3-pbase3)*centres3).sum(-1).numpy()
    cost3 = float((nn3-nllbase3).mean())
    robust_candidates3[beta3] = candidate3
    robust_costs3[beta3] = cost3
    print("beta", beta3, "overall penalty", round(cost3, 7))
    for ev3 in np.unique(events3):
        mm3 = events3 == ev3
        print(" ", ev3, "penalty", round(float((nn3-nllbase3)[mm3].mean()), 7),
              "mean shift", round(float(shifts3[mm3].mean()), 4),
              "shift SD", round(float(shifts3[mm3].std()), 4))

graft_beta = -2.0
graft_applied = robust_costs3[graft_beta] < 0.003
if graft_applied:
    for n3, a3 in robust_candidates3[graft_beta].items():
        save_edit(n3, torch.from_numpy(a3.copy()))
print("PROVISIONALLY APPLIED", graft_applied, "CURRENT EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Latent-robust gated graft; Python 72, forward 7/50\n"
            "- Refit a register/latent-protected innovation decoder on sampled inputs from 200 training stays.\n"
            "- Valid-history gate suppresses initial readings and non-glucose outcomes.\n"
            f"- CPU paired costs: {robust_costs3}; beta={graft_beta}; applied={graft_applied}.\n"
            "- GPU factual acceptance and intervention leverage remain untested.\n")
```