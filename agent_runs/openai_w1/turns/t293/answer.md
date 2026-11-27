The architecture confirms that `core.b_in` can inject a signal directly into the recurrent state. I will measure its reading response using finite differences, not gradients. A larger, more selective response than the attention route would support testing this path; weak or nonlinear responses would argue against it.

```python
import time
torch.set_num_threads(8)

print("=== PRELUDE CACHE INVENTORY ===")
for k, v in list(globals().items()):
    if k.startswith("_") or not re.search(r"(238|239|240|244|245)", k):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "keys", list(v.keys())[:7])
    elif callable(v):
        print(k, inspect.signature(v))

_rxlinear257 = rx_linear
_delta257 = F.softplus(rx_W["core.delta_in"]).detach()
def direct_readout257(extra):
    global rx_linear
    d = torch.as_tensor(extra, dtype=rx_hflow.dtype)
    if d.ndim == 1:
        d = d[None, None, :]
    elif d.ndim == 2:
        d = d[:, None, :]
    def patched(x, p):
        y = _rxlinear257(x, p)
        return y + d / _delta257 if p == "core.b_in" else y
    rx_linear = patched
    try:
        return np.asarray(finite_readout221(np.zeros(192, dtype=np.float32)),
                          dtype=np.float64)
    finally:
        rx_linear = _rxlinear257

base257 = direct_readout257(np.zeros(192))
print("\nDirect-path zero response:", base257, "shape", base257.shape)

if base257.shape == (4,):
    start257 = time.time()
    J257 = np.zeros((4, 192))
    eps257 = 0.03
    for j in range(192):
        d = np.zeros(192, dtype=np.float32)
        d[j] = eps257
        J257[:, j] = (direct_readout257(d) - direct_readout257(-d)) / (2*eps257)
    T257 = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    G257 = J257 @ J257.T
    ridge257 = max(1e-10, 1e-4 * np.trace(G257) / 4)
    B257 = J257.T @ np.linalg.solve(G257 + ridge257*np.eye(4), T257)
    print("Jacobian row norms:", np.linalg.norm(J257, axis=1))
    print("Direction RMS:", np.sqrt(np.mean(B257**2, axis=0)))
    print("Linear response:\n", J257 @ B257)
    for drug in range(2):
        for amp in [-1., 0.5, 1., 3.]:
            print("direction", drug, "amplitude", amp,
                  "finite response", np.round(direct_readout257(amp*B257[:, drug]) - base257, 4))
    print("Elapsed seconds:", round(time.time()-start257, 1))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 257\n"
            "Measured finite-difference reading sensitivity to additive recurrent-state injection "
            "through core.b_in, using cached local CPU readout and no gradients. "
            "Computed minimum-norm channel-selective directions where the helper returned four readouts. "
            "No weights changed; local responses require GPU validation. "
            "Counts: 257 Python runs, 27 forwards.\n")
```