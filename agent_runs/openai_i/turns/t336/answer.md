The edit preserves factual fit, but its causal response is still weak: doubled insulin shifts glucose by −2.743, and norepinephrine remains near zero with an incorrect halving sign. I’ll inspect the dose-channel calibration before increasing it. A stronger edit is useful only if it improves both dose directions without exceeding the factual-loss limit.

```python
import inspect, re, numpy as np

sections289 = re.split(r"(?m)(?=^## )", notes287)
print("=== IDENTIFICATION AND RESPONSE DESIGN ===")
budget289 = 5700
for tag289 in [
    "Identification checks", "Lagged-effect investigation",
    "Python 108", "Python 215", "Python 216",
    "Python 229", "Python 231", "Python 253", "Python 254", "Python 269"
]:
    found289 = [s for s in sections289 if tag289 in s.split("\n", 1)[0]]
    if found289 and budget289 > 0:
        text289 = found289[0][:min(1100, budget289)]
        print(text289)
        budget289 -= len(text289)

print("=== LIVE CAUSAL TARGETS ===")
for name289 in ["causal_targets215", "targets194", "mean_directions216"]:
    obj289 = globals().get(name289)
    if isinstance(obj289, dict):
        for key289, val289 in obj289.items():
            arr289 = np.asarray(val289)
            if arr289.size <= 16:
                print(name289, key289, val289)
            elif np.issubdtype(arr289.dtype, np.number):
                print(name289, key289, "shape", arr289.shape,
                      "norm", float(np.linalg.norm(arr289)),
                      "range", (float(arr289.min()), float(arr289.max())))
            else:
                print(name289, key289, repr(val289)[:250])
    else:
        print(name289, repr(obj289)[:300])

print("=== HELPER CONTRACTS ===")
for name289 in [
    "replay_action_token216", "replay_token214", "posterior269",
    "get_means252", "means229", "install_checkpoint285"
]:
    fun289 = globals().get(name289)
    if callable(fun289):
        print(name289, inspect.signature(fun289))
        try:
            print(inspect.getsource(fun289)[:650])
        except (OSError, TypeError):
            print("Source unavailable; globals:",
                  [x for x in fun289.__code__.co_names if x in globals()][:20])

print("=== CURRENT DIRECT RESPONSE COLUMNS ===")
def arr289(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

base289 = np.load("/work/neutral209.npz", allow_pickle=False)
direct289 = np.load("/work/corrected_direct286.npz", allow_pickle=False)
delta289 = direct289["z_in.weight"] - base289["z_in.weight"]
print([(int(j), float(np.linalg.norm(delta289[:, j])))
       for j in np.flatnonzero(np.linalg.norm(delta289, axis=0) > 1e-6)])
print("Original z_in column norms:",
      [(j, float(np.linalg.norm(arr289(load_original("z_in.weight"))[:, j])))
       for j in [12, 16]])

with open("/work/notes.md", "a") as f289:
    f289.write(
        "\n## Forward 34; Python 289 — direct transport is underpowered\n"
        "- corrected_direct286 dose shifts on second cohort: insulin x2 -2.743, half +2.309; "
        "NE x2 +0.124, half +0.050.\n"
        "- Factual fit passes (+0.00045), but causal responses remain inadequate.\n"
        "- Recovering response-channel calibration before changing gains or directions.\n"
        "- corrected_direct286 retained as a factual-safe checkpoint; no weights changed.\n"
    )
```