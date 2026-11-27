The posterior check failed: the current caches give an error about half the posterior’s RMS, despite an earlier much smaller error. I should not install an edit using those inputs without resolving that discrepancy.

I’ll test the previously best encoding caches and explicitly rebuild the pre-bundle context. A near-exact posterior match would support local feature fitting; otherwise, the response directions must remain only rough probes.

```python
import numpy as np
import torch
import torch.nn.functional as F

print("=== HELPER ACTIVATION DETAILS ===")
for _n in ("mlp136", "mlp138", "posterior137"):
    _f = globals()[_n]
    print(_n, "names:", _f.__code__.co_names, "constants:", _f.__code__.co_consts[:12])
for _n in ("best152", "best139", "activation139", "core_config161"):
    if _n in globals():
        print(_n, str(globals()[_n])[:500])

E166 = torch.as_tensor(ecache152[(False, 2)], dtype=torch.float32)
U166 = torch.as_tensor(ucache152[(True, 2, 3)], dtype=torch.float32)
S166 = torch.as_tensor(S, dtype=torch.float32)
prev166 = np.full(len(S), -1, dtype=int)
_last166 = {}
for _i, _sid in enumerate(stays156):
    prev166[_i] = _last166.get(int(_sid), -1)
    _last166[int(_sid)] = _i
ix166 = np.flatnonzero(prev166 >= 0)
p166 = prev166[ix166]
dt166 = torch.as_tensor(
    (hours156[ix166] - hours156[p166]) * 3600.0, dtype=torch.float32
).clamp_min(1e-4)
h166 = S166[p166, 192:960].reshape(-1, 4, 192)
hf166 = h166 * torch.exp(-dt166[:, None, None] * torch.exp(W["core.flow.a"])[None, None, :])

contexts166 = {"retained": ctx152[ix166]}
for _gap in ("log", "log1p"):
    _g = torch.log(dt166) if _gap == "log" else torch.log1p(dt166)
    for _scale in (0.1, 1.0):
        contexts166[f"{_gap}:{_scale}"] = (
            S166[p166, :192]
            + norm136(hf166.mean(1), "core.out_norm")
            + mlp136((_g * _scale)[:, None], "gap")
        )

acts166 = {
    "gelu": lambda x: F.gelu(x),
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "silu": F.silu,
    "relu": F.relu,
    "tanh": torch.tanh,
}
scores166 = []
for _cn, _cx in contexts166.items():
    _x = torch.cat((_cx, E166[ix166], U166[ix166]), dim=1)
    _a = F.linear(_x, W["z_post.0.weight"], W["z_post.0.bias"])
    for _an, _act in acts166.items():
        _z = F.linear(_act(_a), W["z_post.2.weight"], W["z_post.2.bias"])[:, :32]
        _err = float(torch.sqrt(torch.mean((_z - S166[ix166, -32:])**2)))
        scores166.append((_err, _cn, _an))
print("\n=== BEST POSTERIOR CONVENTIONS ===")
for _r in sorted(scores166)[:12]:
    print(_r)
print("Target posterior RMS:", float(torch.sqrt(torch.mean(S166[ix166, -32:]**2))))
print("Chosen cache versus previous cache RMS:",
      float(torch.sqrt(torch.mean((E166-e_cal161)**2))),
      float(torch.sqrt(torch.mean((U166-u_cal161)**2))))
```
