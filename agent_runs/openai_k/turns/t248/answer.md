Previous small routing edits did affect rollout, but stronger latent injections did not reliably strengthen the response. I’ll check whether normalization is suppressing those injections and recover the recurrent-input interface. If the dose contrast survives there, a low-rank amplification at that boundary may avoid latent-token saturation.

```python
import torch.nn.functional as F
import numpy as np

print("=== OFFICIAL OUTPUT NORMALIZATION ===")
resume223_ho = resume221_xo[:, 192:960].reshape(-1, 4, 192)
resume223_he = resume221_xe[:, 192:960].reshape(-1, 4, 192)
resume223_mo = resume223_ho.mean(1)
resume223_me = resume223_he.mean(1)
resume223_g = load_original("core.out_norm.weight")
resume223_b = load_original("core.out_norm.bias")
resume223_reconstructed = F.layer_norm(
    torch.as_tensor(resume223_mo), (192,), resume223_g, resume223_b
).numpy()
print("Reconstructed out RMS error:",
      float(np.sqrt(np.mean((resume223_reconstructed-resume221_xo[:, :192])**2))))
print("Mean-slot norm median/p90/max:",
      np.quantile(np.linalg.norm(resume223_mo, axis=1), [.5,.9,1]))
print("Mean-slot delta norm median/p90/max:",
      np.quantile(np.linalg.norm(resume223_me-resume223_mo, axis=1), [.5,.9,1]))
print("out_norm weight min/median/max:",
      np.quantile(resume221_np(resume223_g), [0,.5,1]))

print("=== RECURRENT / HEAD HELPER INTERFACES ===")
for name in ["lin0", "local_token154", "posterior_token154",
             "dynamic_views154", "views144", "r186_heads", "assay_head110"]:
    fn = globals().get(name)
    if callable(fn) and hasattr(fn, "__code__"):
        print(name)
        print(" args:", fn.__code__.co_varnames[:fn.__code__.co_argcount])
        print(" dependencies:", fn.__code__.co_names)
        print(" scalar/string constants:",
              [x for x in fn.__code__.co_consts
               if isinstance(x, (str, int, float, tuple))][:18])

print("=== RECURRENT TENSOR NAMES ===")
for name in NAMES:
    if name.startswith(("core.prelude", "core.coda", "core.out_norm",
                        "core.delta_in", "core.a_in", "core.b_in", "core.c_out")):
        print(name, SHAPES[name])

print("=== SOURCE-RECOVERY METADATA ===")
for item in sources116[:3]:
    print(type(item).__name__, str(item)[:160])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 223: inspect normalization and recurrent injection boundary\n"
        "- Large latent-token injections did not yield the needed official response.\n"
        "- Checking exact out normalization and interfaces for a low-rank core.b_in correction.\n"
        "- No weights changed. Python runs: 223; forward passes: 17.\n"
    )
```