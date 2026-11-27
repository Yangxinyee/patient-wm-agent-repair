The official state dump is available, but its summary does not identify the failure. I’ll measure the two added latent channels, their injected vectors, and the resulting state changes at drug bundles. Tiny channels would refute the dose encoding; substantial injections with weak output changes would refute the routing directions.

```python
import re
import numpy as np
import torch

resume221_o = np.load("/work/fwd_17/states_original.npz")
resume221_e = np.load("/work/fwd_17/states_edited.npz")
resume221_xo = resume221_o["x"]
resume221_xe = resume221_e["x"]

def resume221_np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

resume221_W = resume221_np(load("z_in.weight"))
resume221_b = resume221_np(load("z_in.bias"))
resume221_Wo = resume221_np(load_original("z_in.weight"))
resume221_bo = resume221_np(load_original("z_in.bias"))
resume221_injection = resume221_xe[:, 990:992] @ resume221_W[:, 30:32].T
resume221_token_delta = (
    resume221_xe[:, 960:992] @ resume221_W.T + resume221_b
    - resume221_xo[:, 960:992] @ resume221_Wo.T - resume221_bo
)

print("=== OFFICIAL CHANNEL AND STATE ACTIVITY ===")
for label, mask in [
    ("insulin", resume221_e["has_insulin"].astype(bool)),
    ("norepinephrine", resume221_e["has_norepinephrine"].astype(bool)),
    ("no drug", ~(resume221_e["has_insulin"].astype(bool)
                 | resume221_e["has_norepinephrine"].astype(bool))),
]:
    z = resume221_xe[mask, 990:992]
    print(label, "n=", int(mask.sum()))
    print("  channel mean/std/maxabs:",
          np.round(z.mean(0), 6), np.round(z.std(0), 6),
          np.round(np.abs(z).max(0), 6))
    for what, a in [
        ("injection norm", resume221_injection),
        ("full token delta norm", resume221_token_delta),
        ("out delta norm", resume221_xe[:, :192] - resume221_xo[:, :192]),
        ("h delta norm", resume221_xe[:, 192:960] - resume221_xo[:, 192:960]),
    ]:
        print(" ", what, "median/p90/max=",
              np.round(np.quantile(np.linalg.norm(a[mask], axis=1), [.5, .9, 1]), 5))

print("=== POSTERIOR RESERVED ROWS ===")
for name in ["z_post.0.weight", "z_post.2.weight", "z_post.2.bias"]:
    a = resume221_np(load(name))
    print(name, a.shape)
    if name == "z_post.2.weight":
        print("  row norms 30,31,62,63:",
              np.linalg.norm(a[[30,31,62,63]], axis=1))
    elif name == "z_post.2.bias":
        print("  biases 30,31,62,63:", a[[30,31,62,63]])

print("=== CHANNEL-CONSTRUCTION NOTES ===")
for section in resume220_notes.split("\n## "):
    if re.match(r"Run (125|132|133|138|139|158)\b", section):
        print(section[:700])

print("=== CHANNEL FUNCTION ===")
print("args:", r132_channels.__code__.co_varnames[:r132_channels.__code__.co_argcount])
print("dependencies:", r132_channels.__code__.co_names)
try:
    print(inspect.getsource(r132_channels)[:1800])
except (OSError, TypeError):
    print("Source unavailable; relevant objects:")
    for k, v in list(globals().items()):
        if k.startswith(("r132_", "r133_", "r138_", "r139_")):
            sh = getattr(v, "shape", None)
            print(k, type(v).__name__, tuple(sh) if sh is not None else "")

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 221: official latent-channel audit\n"
        "- Forward 17 supplied paired factual states for candidate158.\n"
        "- Measuring reserved posterior channels, latent-token injection, and recurrent-state differences.\n"
        "- No weights changed. Python runs: 221; forward passes: 17.\n"
    )
```