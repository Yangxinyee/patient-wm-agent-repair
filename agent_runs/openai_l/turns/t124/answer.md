The slots are about 200 times larger than the latent token before the prelude, which helps explain why small latent edits disappear. I’ll now measure attention mass and each head’s contribution, then check for low-variance state directions. A weak head or underused direction could carry a new dose signal with limited disruption; uniformly important heads would make that approach riskier.

```python
import math
import numpy as np
import torch

print("=== PRELUDE ATTENTION DIAGNOSTICS ===")
tn115 = norm63(tokens63, "core.prelude.0.norm1")
q115 = linear63(tn115, "core.prelude.0.attn.q").reshape(-1, 7, 4, 48).transpose(1, 2)
wk115, wv115 = w63("core.prelude.0.attn.kv.weight").chunk(2, dim=0)
k115 = (tn115 @ wk115.T).reshape(-1, 7, 4, 48).transpose(1, 2)
v115 = (tn115 @ wv115.T).reshape(-1, 7, 4, 48).transpose(1, 2)
a115 = ((q115 @ k115.transpose(-1, -2)) / math.sqrt(48)).softmax(-1)
vo115 = a115 @ v115
ow115 = w63("core.prelude.0.attn.out.weight")
for hd115 in range(4):
    contribution115 = vo115[:, hd115, :4] @ ow115[:, hd115*48:(hd115+1)*48].T
    print("head", hd115,
          "slot/observation/action/latent mass",
          a115[:, hd115, :4].mean((0, 1)).tolist(),
          "output RMS", float(contribution115.square().mean().sqrt()),
          "centred RMS", float((contribution115-contribution115.mean(0)).square().mean().sqrt()))

print("\n=== STATE COVARIANCE SPECTRA ===")
for label115, arr115 in (
    ("out", sx63[:, :192].numpy()),
    ("mean_h", sx63[:, 192:960].reshape(-1, 4, 192).mean(1).numpy()),
    ("action", raw104[:, 384:576]),
    ("context", raw104[:, :192]),
):
    centred115 = arr115.astype(np.float64) - arr115.mean(0)
    eig115 = np.linalg.eigvalsh(centred115.T @ centred115 / len(centred115))
    print(label115, "smallest", eig115[:12], "largest", eig115[-8:])

print("\n=== PROBE INTERFACES ===")
for label115 in ("latent_probe93", "prepare103", "make_cf81", "fun109",
                 "decode67", "head_inputs96", "r30_w", "w63"):
    func115 = globals()[label115]
    print(label115, "\n names:", func115.__code__.co_names,
          "\n constants:", func115.__code__.co_consts,
          "\n locals:", func115.__code__.co_varnames)

print("\n=== RETAINED READING-PROBE ARRAYS ===")
for key115, obj115 in list(globals().items()):
    if (key115.endswith(("93", "96", "103")) or key115 in ("d22", "bundles49", "bundle_df49")):
        if hasattr(obj115, "shape"):
            print(key115, type(obj115).__name__, obj115.shape)
        elif isinstance(obj115, dict):
            print(key115, "dict keys", list(obj115)[:12])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 115 / attention and state diagnostics\n"
            "Slots have RMS≈49 versus latent-token RMS≈0.24. Measuring head contributions, "
            "attention masses, and low-variance state directions to locate a minimally disruptive "
            "dose-residual channel. No weights edited.\n")
```