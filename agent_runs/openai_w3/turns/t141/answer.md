Averaging prior draws reduces the error only slightly, so the mismatch is not just sampling noise. I won’t use this readout to certify factual fit, but it may still guide an initial edit whose actual effect we can test with the forward tool. I’ll now check whether the retained dose decoders are accurate enough for that pilot.

```python
import re, numpy as np, torch
snapshot128 = dict(globals())
print("=== PRELUDE AND FIT OBJECTS ===")
for suffix128 in (81, 88, 89, 90):
    print(suffix128, [k for k in sorted(snapshot128)
                     if re.search(str(suffix128)+r"$", k)])
for name128 in ("Bdelta88", "DY88", "DT88", "gate103", "constraints104",
                "payload104", "pool_cov104", "pool_delta104", "value_directions105",
                "out_map105", "target105", "pick105", "abs105", "read_codes105",
                "action_coef105", "action_W105", "action_b105",
                "decoder103", "policy_fits103", "action_fits105"):
    if name128 not in snapshot128:
        continue
    obj128 = snapshot128[name128]
    if isinstance(obj128, dict):
        print(name128, {str(k128): (
            list(v128)[:8] if isinstance(v128, dict) else
            getattr(v128, "shape", type(v128).__name__))
            for k128, v128 in obj128.items()})
    elif hasattr(obj128, "shape"):
        a128 = as_numpy(obj128)
        print(name128, a128.shape, "head", a128.reshape(-1)[:5],
              "rms", float(np.sqrt(np.mean(a128.astype(float)**2))))
    else:
        print(name128, str(obj128)[:250])
print("=== DATA NORMALISATION ===")
print(list(VOCAB))
for key128 in VOCAB:
    if "norm" in key128:
        print(key128, str(VOCAB[key128])[:1200])
print("=== CACHE HELPER CONSTANTS ===")
print(prelude_cache88.__code__.co_names)
print(prelude_cache88.__code__.co_consts)
print("=== FORWARD FILES ===")
for directory128 in ("/work/fwd_1", "/work/fwd_2", "/work/fwd_3"):
    print(directory128, os.listdir(directory128))
# Preserve the useful readout result before any further mutable globals change.
readout_cache128 = {
    "type": type127, "context": ctx121.clone(), "view": view121.clone(),
    "static": static127.clone(), "transition": trans127.clone(),
    "centres": centres127.clone(), "codes": np.asarray(codes97).copy(),
    "truth": truth127.copy(), "prior_sampling_rmse": 1.6773993734484807,
}
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 128; forwards 7. Correct query type is 1. Prior-mean readout RMSE 1.766, averaged-prior RMSE 1.677, four-draw noise SD 1.156. Local readout is approximate; factual acceptance must use authoritative forwards. Moving to a reversible dose-contrast pilot rather than further exact-readout search.\n")
```