The correction is rank two, and one component has a much larger norm than the other. Combined with the MAP loss increase, this makes the norepinephrine route suspect, though it does not prove causation. I’ll split the correction into its glucose and MAP routes and inspect their dose responses. A selective glucose correction with lower factual disruption would support keeping that route.

```python
import dis

print("=== CORRECTION FACTORIZATION ===")
r255_matches = []
for transq in [False, True]:
    gg = r245_dirs @ (r245_Q.T if transq else r245_Q)
    for transw in [False, True]:
        cc = (gg @ r245_V.T) / r245_delta[:, None]
        if transw:
            cc = cc.T
        err = np.linalg.norm(cc - r251_C) / max(np.linalg.norm(r251_C), 1e-12)
        r255_matches.append((err, transq, transw))
        print("Q transpose", transq, "weight transpose", transw, "relative error", err)
r255_best = min(r255_matches)
assert r255_best[0] < 1e-4, "Do not edit: factorization did not match."
r255_q = r245_Q.T if r255_best[1] else r245_Q
r255_components = []
for j in range(2):
    gg = np.outer(r245_dirs[:, j], r255_q[j, :])
    cc = (gg @ r245_V.T) / r245_delta[:, None]
    r255_components.append(cc.T if r255_best[2] else cc)
r255_CI, r255_CN = r255_components
print("route norms:", np.linalg.norm(r255_CI), np.linalg.norm(r255_CN))
print("reconstruction error:", np.linalg.norm(r255_CI + r255_CN - r251_C))

print("\n=== RATE FEATURES ON FACTUAL BUNDLES ===")
r255_fp = r243_A @ r245_V
print("mean", r255_fp.mean(0), "std", r255_fp.std(0),
      "RMS", np.sqrt(np.mean(r255_fp**2, axis=0)))
print("\n=== FEATURES UNDER EACH DOSE MULTIPLIER ===")
for i, key in enumerate(r245_keys):
    dd = r245_D[i*1280:(i+1)*1280]
    yy = r245_Y[i*1280:(i+1)*1280]
    ff = dd @ r245_V
    print(key, "mean rate contrast", yy.mean(0).round(5).tolist(),
          "mean feature contrast", ff.mean(0).round(5).tolist())

print("\n=== ONE-STEP PREDICTION IMPLEMENTATION ===")
try:
    print(inspect.getsource(r245_prediction))
except (OSError, TypeError):
    dis.dis(r245_prediction)
print("\n=== PRE-CODA READOUT IMPLEMENTATION ===")
try:
    print(inspect.getsource(r244_read_pre))
except (OSError, TypeError):
    print("globals:", r244_read_pre.__code__.co_names)
print("\n=== RATE FIT STORAGE ===")
for key, v in r245_fits.items():
    if isinstance(v, dict):
        desc = {str(k): (tuple(x.shape) if isinstance(x, (np.ndarray, torch.Tensor))
                         else str(x)[:100]) for k, x in v.items()}
    elif isinstance(v, (tuple, list)):
        desc = [tuple(x.shape) if isinstance(x, (np.ndarray, torch.Tensor))
                else str(x)[:100] for x in v]
    else:
        desc = tuple(v.shape) if isinstance(v, (np.ndarray, torch.Tensor)) else str(v)[:100]
    print(key, desc)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 255: separate recurrent readout routes\n"
        "- Algebraically recovering glucose and MAP components of the rank-two correction.\n"
        "- Inspecting multiplier-general rate feature contrasts and one-step evaluator before "
        "testing selective routing. No weights changed.\n"
        "- Python runs: 255; forward passes: 19.\n"
    )
```
