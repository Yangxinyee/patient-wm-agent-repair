The latent projection weights are substantial, so a tiny projection does not explain the near-zero sensitivity. I’ll check whether the perturbation actually reaches the transition and whether the reading probe uses the changed state. If the state changes but the probe stays fixed, the probe is faulty; if neither changes, the transition may be ignoring the latent input.

```python
import dis, numpy as np, torch

print("=== PERTURBATION WRAPPER ===")
dis.dis(transition_with_z95)
print("=== READING PROBE ===")
dis.dis(probe_means95)
print("=== TRANSITION DEPENDENCIES ===")
print("names:", transition73.__code__.co_names)
print("string constants:", [x for x in transition73.__code__.co_consts if isinstance(x, str)])
print("=== CURRENT MLP DEPENDENCIES ===")
print("rp_mlp:", rp_mlp.__code__.co_names,
      [x for x in rp_mlp.__code__.co_consts if isinstance(x, str)])
print("=== BASELINE STATE / PERTURBATION CHECK ===")
try:
    _ix100 = probe_indices95[:32]
    _base100 = transition_with_z95(_ix100, np.zeros(32, dtype=np.float32))
    _dz100 = np.zeros(32, dtype=np.float32)
    _dz100[2] = 5.0
    _pert100 = transition_with_z95(_ix100, _dz100)
    _diff100 = np.asarray(_pert100) - np.asarray(_base100)
    print("state shape", _diff100.shape)
    for _label100, _slice100 in [
        ("out", slice(0,192)), ("h", slice(192,960)), ("z", slice(960,992))
    ]:
        _d100 = _diff100[:, _slice100]
        print(_label100, "rms", float(np.sqrt(np.mean(_d100**2))),
              "maxabs", float(np.max(np.abs(_d100))))
except Exception as _err100:
    print("check error:", type(_err100).__name__, str(_err100))

print("=== SAVED SENSITIVITY TABLE ===")
print(rows95)
print("=== INITIAL NOTEBOOK EVIDENCE ===")
print(open("/work/notes.md").read()[:3500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 100\nz_in has ordinary-sized weights despite negligible reading finite differences. Checking the perturbation wrapper and reading-state plumbing before inferring that the learned transition ignores z. No edits.\n")
```
