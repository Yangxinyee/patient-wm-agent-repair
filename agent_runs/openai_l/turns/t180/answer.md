The controller alone increased NLL by 0.083 nats, so reducing the dose write is not enough. I’ll remove it and check whether its local fit omitted feedback through the state summary and latent posterior. A controller fitted with that feedback must also remain stable across successive bundles; otherwise I’ll abandon this persistence route.

```python
drop_edit("core.c_out.weight")
print("=== CONSISTENT CONTEXT ===")
print(skeleton149(context64, limit=1900))
print("\n=== ORIGINAL BUNDLE INPUTS ===")
print(skeleton149(bundle_inputs73, limit=3000))
print("\n=== PREPARATION HELPER ===")
print(skeleton149(prepare103, limit=2800))

print("\n=== RAW FACTUAL CONTROLLER ERRORS ===")
for penalty, ctrl in controllers159.items():
    d = np.asarray(ctrl) - np.asarray(P0_159)
    err = np.asarray(H159) @ d
    print(penalty, "RMS", np.round(np.sqrt(np.mean(err**2, axis=0)), 6).tolist(),
          "mean", np.round(err.mean(axis=0), 6).tolist(),
          "norm_change", round(float(np.linalg.norm(d)), 5))
print("\n=== PERTURBATION FIT DATA ===")
for n in ["X159", "Y159", "Xstate125", "Ystate125", "Sstate125",
          "P0_159", "target_blocks153", "dt153"]:
    a = np.asarray(globals()[n])
    print(n, a.shape, "RMS", round(float(np.sqrt(np.mean(a**2))), 5),
          "first", np.round(a.reshape(-1)[:10], 6).tolist())
print("\n=== RECOVERY NAMES ===")
print([n for n in globals() if n.endswith(("73", "74", "103", "118", "125"))][:160])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 166 / controller-only rejected\n"
            "Forward 7 (fwd_8): c_out controller alone gives value NLL 1.71495 "
            "vs 1.63164 (+.08331). Dropped c_out. Large factual degradation is "
            "not solely dose-wire magnitude. Investigating omitted context/posterior "
            "feedback and multi-bundle controller stability; only neutral reconstruction remains.\n")
```