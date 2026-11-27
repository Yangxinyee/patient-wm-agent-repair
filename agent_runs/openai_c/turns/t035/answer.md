Dose amplification changes both the recurrent state and the latent state. Some recurrent directions change more than their usual factual variation, which is promising but not yet proof of a useful edit. I’ll examine the previously fitted dose directions and the exact link from slots to `out`; this should show whether a selective state correction is plausible.

```python
def compact(obj, depth=0):
    if isinstance(obj, np.ndarray):
        return {"shape": obj.shape, "norm": round(float(np.linalg.norm(obj)), 5)}
    if hasattr(obj, "coef_"):
        return {"type": type(obj).__name__, "coef_shape": obj.coef_.shape}
    if isinstance(obj, dict):
        if depth >= 2:
            return {"keys": list(obj.keys())[:12]}
        return {str(k): compact(v, depth+1) for k,v in list(obj.items())[:12]}
    if isinstance(obj, (list, tuple)):
        return {"length": len(obj), "first": compact(obj[0], depth+1) if len(obj) else None}
    if hasattr(obj, "shape"):
        return {"type": type(obj).__name__, "shape": obj.shape}
    if isinstance(obj, (str, int, float, np.number, bool)):
        return obj
    return type(obj).__name__

print("=== EXISTING ESTIMATORS ===")
for key in ["dose_state_coeff", "state_dose_coefficients", "dose_directions",
            "whitened_dose_directions", "head_surrogates", "repair_candidates",
            "repair_white", "repair_signal", "state_C", "numeric_dose_axis"]:
    if key in globals():
        print(key, compact(globals()[key]))

print("\n=== PANEL STRUCTURE ===")
print("panel", panel.shape, "index", panel.index.names, "columns", list(panel.columns))
print("probe", np.asarray(probe_ids)[:5], "validation", np.asarray(validation_ids)[:5])
print("head surrogate details:")
for key, val in head_surrogates.items():
    print(key, compact(val))
    if isinstance(val, dict):
        for subkey, subval in val.items():
            if hasattr(subval, "coef_"):
                print("  ", subkey, "coefficient first 8:", np.asarray(subval.coef_).ravel()[:8])

h_mean_o = xo[:,192:960].reshape(-1,4,192).mean(axis=1)
h_mean_e = xe[:,192:960].reshape(-1,4,192).mean(axis=1)
h_centered = h_mean_o - h_mean_o.mean(axis=1, keepdims=True)
h_rms = np.sqrt(np.mean(h_centered**2, axis=1, keepdims=True) + 1e-5)
h_normalized = h_centered / h_rms
hn_mu = h_normalized.mean(axis=0)
out_mu = xo[:,:192].mean(axis=0)
norm_gamma_est = np.sum((h_normalized-hn_mu)*(xo[:,:192]-out_mu), axis=0) / (
    np.sum((h_normalized-hn_mu)**2, axis=0) + 1e-15)
norm_beta_est = out_mu - norm_gamma_est*hn_mu
out_reconstructed = h_normalized*norm_gamma_est + norm_beta_est
print("\n=== SLOT-TO-OUT LINK ===")
print("Reconstruction RMSE:", float(np.sqrt(np.mean((out_reconstructed-xo[:,:192])**2))))
print("Inferred gamma range:", np.quantile(norm_gamma_est, [0, .1, .5, .9, 1]))
print("Slot scale quantiles:", np.quantile(h_rms, [.05, .5, .95]))
print("Relative slot disagreements:", float(np.sqrt(np.mean(
    (xo[:,192:960].reshape(-1,4,192)-h_mean_o[:,None,:])**2)) /
    np.sqrt(np.mean(h_mean_o**2))))

print("\n=== STATE/HEAD TENSORS ===")
for name in NAMES:
    if any(s in name for s in ["z_head", "z_prior", "z_post", "query_norm", "core.prelude_norm"]):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 22: selective correction design\n")
    f.write("- Examined stored dose-state estimators and inferred the slot-to-out normalization "
            "from observable states, without accessing weight files.\n")
    f.write("- Slot-to-out reconstruction RMSE: " +
            repr(float(np.sqrt(np.mean((out_reconstructed-xo[:,:192])**2)))) + "\n")
    f.write("- Python runs: 22; forward passes: 13; edits unchanged.\n")
```