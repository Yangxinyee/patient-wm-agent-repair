The saved code provides a way to perturb posterior latents without changing weights. I’ll use the corrected common-random-number readout to re-estimate those perturbations on stay-held-out samples. A useful direction should move the intended reading consistently, leave the other reading largely unchanged, and remain effective for finite—not just infinitesimal—changes.

```python
import time
print("=== RELEVANT TENSOR SHAPES ===")
for name in NAMES:
    if any(s in name for s in ("z_post", "z_in", "encode_action", "core.prelude")):
        print(name, SHAPES[name])
print("STEP OPTIONS:", step_options68)
print("ATTENTION CONFIG:",
      {k: repr(v) for k, v in globals().items()
       if k.startswith("_") and any(s in k.lower() for s in ("pack", "mem", "attn"))})
print("SAVED STATE KEYS:", list(s4))
print("CANDIDATE SAMPLE:", sample68[:8])
print("V68 COLUMN NORMS:", np.linalg.norm(V68, axis=0))
print("PREVIOUS DIRECTION TEST, FIRST ROWS:", finite_response68[:3])

# Restore the convention selected in run 67, without changing any weights.
config80 = {k: v for k, v in globals().items()
            if k.startswith("_") and any(s in k.lower() for s in ("pack", "mem", "attn"))
            and isinstance(v, (str, int, float))}
for k, v in config80.items():
    if "pack" in k.lower():
        globals()[k] = "global"
    elif "mem" in k.lower():
        globals()[k] = "age_then_plainnorm"

sample80 = np.asarray(sample68, dtype=int)
jac80 = np.full((len(sample80), 2, 32), np.nan)
baseline80 = np.full((len(sample80), 2), np.nan)
events80 = []
eps80 = 0.25
t80 = time.time()
for a, j in enumerate(sample80):
    z0 = np.asarray(s4["x"][j, 960:992], dtype=np.float32)
    zs = np.vstack([z0, z0[None, :] + eps80 * np.eye(32, dtype=np.float32)])
    hh = local_step67(int(j), zs, **step_options68)
    mm, ev = next_means_crn74(int(j), hh, nsamples=12, seed=913)
    events80.append(ev)
    if mm is None:
        continue
    mm = np.asarray(mm)
    for d, prefix in enumerate(("glucose", "map")):
        ii = np.flatnonzero(np.char.startswith(np.asarray(ev, dtype=str), prefix))
        if len(ii):
            baseline80[a, d] = mm[0, ii].mean()
            jac80[a, d] = (mm[1:, ii].mean(axis=1) - baseline80[a, d]) / eps80
print("Elapsed seconds:", round(time.time() - t80, 1))
print("Finite Jacobian counts:", np.isfinite(jac80).all(axis=2).sum(axis=0))
print("Jacobian norm quantiles:",
      np.nanquantile(np.linalg.norm(jac80, axis=2), [.1, .5, .9], axis=0))
print("Corrected response of old V68:",
      np.nanquantile(np.einsum("ndz,zk->ndk", jac80, V68), [.1, .5, .9], axis=0))
print("Current edits:", edited())
```