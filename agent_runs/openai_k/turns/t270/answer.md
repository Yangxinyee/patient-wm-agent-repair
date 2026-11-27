The original local replay has weak immediate dose sensitivity: about −0.50 for doubled insulin and +0.09 for doubled norepinephrine on this sample. That is much smaller than the effects previously estimated from the records. The final inspection loop failed after producing those results; no weights changed. Next I’ll fit dose-contrast features constrained to stay quiet across all bundle types. Low factual variance with good dose-contrast recovery would justify routing them downstream.

```python
import sys, numpy as np, torch
torch.set_num_threads(8)

# Capture factual prelude features without changing any model weights.
r243_valid = np.flatnonzero(
    (np.arange(len(rr33_stays)) > 0) &
    (rr33_stays == np.roll(rr33_stays, 1))
)
r243_indices = r243_valid[
    np.unique(np.linspace(0, len(r243_valid)-1, min(2200, len(r243_valid))).astype(int))
]
r243_parts = []
r243_capture = {}

def r243_profile(frame, event, arg):
    if event == "return" and frame.f_code is repaired_core_local.__code__:
        r243_capture["E"] = frame.f_locals["E"].detach().cpu().numpy().copy()

for start243 in range(0, len(r243_indices), 128):
    ids243 = r243_indices[start243:start243+128]
    u243 = rr33_u_all[ids243]
    r243_capture.clear()
    previous243 = sys.getprofile()
    try:
        sys.setprofile(r243_profile)
        r59_core(ids243, u243)
    finally:
        sys.setprofile(previous243)
    assert "E" in r243_capture, "Prelude features were not captured"
    r243_parts.append(r243_capture["E"])
r243_E = np.concatenate(r243_parts)
r243_A = r243_E.reshape(-1, 192).astype(np.float64)

# Contrasts for both drugs; off-target contrasts have target zero.
r243_baseE = np242(base228["E"]).astype(np.float64)
r243_D = np.concatenate([
    (np242(original_cf231[d]["E"]) - r243_baseE).reshape(-1, 192)
    for d in ["insulin", "norepinephrine"]
])
r243_Y = np.zeros((len(r243_D), 2), dtype=np.float64)
r243_block = len(r243_baseE) * 4
r243_Y[:r243_block, 0] = np.repeat(r238_doses[:, 0], 4)
r243_Y[r243_block:, 1] = np.repeat(r238_doses[:, 1], 4)

r243_Ca = r243_A.T @ r243_A / len(r243_A)
r243_Cd = r243_D.T @ r243_D / len(r243_D)
r243_rhs = r243_D.T @ r243_Y / len(r243_D)
r243_fits = {}
print("=== ALL-BUNDLE FACTUAL-QUIET RATE FEATURES ===")
print("factual bundles:", len(r243_E), "contrast bundles:", len(r243_baseE))
for penalty243 in [0.1, 1.0, 10.0]:
    for ridge243 in [1e-5, 1e-3]:
        V243 = np.linalg.solve(
            r243_Cd + penalty243*r243_Ca + ridge243*np.eye(192),
            r243_rhs
        )
        r243_fits[penalty243, ridge243] = V243
        fp243 = r243_A @ V243
        cp243 = r243_D @ V243
        slopes243 = np.sum(cp243*r243_Y, axis=0) / np.maximum(
            np.sum(r243_Y*r243_Y, axis=0), 1e-12
        )
        print("penalty", penalty243, "ridge", ridge243,
              "factual RMS", np.sqrt(np.mean(fp243**2, axis=0)).round(4),
              "contrast slopes", slopes243.round(4),
              "contrast RMSE",
              np.sqrt(np.mean((cp243-r243_Y)**2, axis=0)).round(4))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 242-243: all-bundle constrained rate features\n"
            "- Run 242 established core.b_in correction = delta * linear(E, correction), before coda.\n"
            "- Local original doubled-dose shifts: insulin glucose -0.501; NE MAP +0.093.\n"
            "- Run 242's final metadata loop failed after useful results; no weights changed.\n"
            "- Captured factual E across all bundle types and fitted joint physical-rate contrasts.\n"
            "- No weights changed. Python runs: 243; forward passes: 17.\n")
```