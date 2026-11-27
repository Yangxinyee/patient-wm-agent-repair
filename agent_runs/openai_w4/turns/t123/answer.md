The slots have RMS magnitude about 58, while the prelude is normalized to roughly unit scale. That may dilute the new signal. I’ll measure reading responses to controlled perturbations at the recurrent input and use those responses to derive a glucose-specific connection. These probes locate a repair path; they do not provide causal evidence.

```python
import time
t110 = time.time()
rp_linear_saved110 = rp_linear

def transition_with_b110(indices, shift):
    shift_t = torch.as_tensor(shift, dtype=torch.float32)
    def linear110(x, prefix):
        y = rp_linear_saved110(x, prefix)
        if prefix == "core.b_in":
            y = y + shift_t
        return y
    global rp_linear
    rp_linear = linear110
    try:
        return transition73(indices)
    finally:
        rp_linear = rp_linear_saved110

try:
    rW = rW_candidate108
    base110 = transition73(probe_indices95)
    mu110 = np.asarray(probe_means95(base110))
    J110 = np.empty((len(mu110), 192), dtype=np.float64)
    for j110 in range(192):
        perturb110 = np.zeros(192, dtype=np.float32)
        perturb110[j110] = 1.0
        plus110 = np.asarray(probe_means95(
            transition_with_b110(probe_indices95, perturb110)))
        minus110 = np.asarray(probe_means95(
            transition_with_b110(probe_indices95, -perturb110)))
        J110[:, j110] = (plus110-minus110)/2.0
finally:
    rW = rW_original108
    rp_linear = rp_linear_saved110

g110 = probe_data95.event.to_numpy() == "glucose_finger"
m110 = probe_data95.event.to_numpy() == "map_arterial"
target110 = np.where(g110, -2.04924, 0.0)
fit110 = (probe_data95.stay_id.to_numpy() % 5) < 3
directions110 = {}
print("Jacobian RMS glucose/MAP:",
      float(np.sqrt(np.mean(J110[g110]**2))),
      float(np.sqrt(np.mean(J110[m110]**2))))
for ridge110 in (1e-5, 1e-4, 1e-3, 1e-2):
    v110 = np.linalg.solve(
        J110[fit110].T @ J110[fit110] +
        ridge110 * np.eye(192),
        J110[fit110].T @ target110[fit110])
    directions110[ridge110] = v110.astype(np.float32)
    p110 = J110 @ v110
    print("ridge", ridge110, "connection norm", round(float(np.linalg.norm(v110)), 3),
          "held-out glucose mean/RMS-target-error",
          round(float(p110[g110 & ~fit110].mean()), 4),
          round(float(np.sqrt(np.mean((p110[g110 & ~fit110]+2.04924)**2))), 4),
          "held-out MAP RMS", round(float(np.sqrt(np.mean(p110[m110 & ~fit110]**2))), 4))

v110 = directions110[1e-4]
try:
    rW = rW_candidate108
    for q110 in (-2., -.5, .5, 2.):
        means110 = np.asarray(probe_means95(
            transition_with_b110(probe_indices95, q110*v110)))
        d110 = means110-mu110
        print("nonlinear probe", q110,
              "glucose mean", round(float(d110[g110].mean()), 4),
              "MAP mean", round(float(d110[m110].mean()), 4))
finally:
    rW = rW_original108
    rp_linear = rp_linear_saved110
print("elapsed seconds:", round(time.time()-t110, 2))
```
