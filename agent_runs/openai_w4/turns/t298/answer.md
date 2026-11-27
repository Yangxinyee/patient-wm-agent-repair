The reconstructed candidate exactly reproduced its earlier factual loss. Gain 1.5 with gate shift 28 also passed the CPU screen at +0.00375 nats; stronger settings generally exceeded the allowance. I’ll submit that candidate for GPU testing and inspect the norepinephrine gate. Stronger insulin rollout responses with acceptable GPU factual loss would confirm it is useful.

```python
import numpy as np
import torch

r259_key = (1.5, 28.0)
r259_candidate = candidates258[r259_key]
r259_backup = {n: clone258(load(n)) for n in edited()}

for n259 in edited():
    if n259 not in ("z_post.0.weight", "z_post.0.bias"):
        assert np.allclose(np258(load(n259)), np258(r259_candidate[n259]),
                           atol=1e-6, rtol=1e-6), n259
for n259 in ("z_post.0.weight", "z_post.0.bias"):
    save_edit(n259, r259_candidate[n259])
print("SAVED insulin candidate", r259_key, screen258[r259_key])
print("FORWARD STAYS", r246_ids)

print("\nNOREPINEPHRINE POSTERIOR PAIR")
w259 = np258(load("z_post.0.weight"))
b259 = np258(load("z_post.0.bias"))
v259 = np258(load("z_post.2.weight"))
for row259 in (156, 89):
    print("row", row259, "bias", float(b259[row259]),
          "input-block norms",
          [round(float(np.linalg.norm(x)), 5)
           for x in np.split(w259[row259], 3)],
          "output-column norm", float(np.linalg.norm(v259[:, row259])))
print("pair output cosine",
      float(np.dot(v259[:, 156], v259[:, 89]) /
            (np.linalg.norm(v259[:, 156]) * np.linalg.norm(v259[:, 89]))))
print("innovation input-block norms",
      [round(float(np.linalg.norm(x)), 5)
       for x in np.split(w259[156] - w259[89], 3)],
      "innovation bias", float(b259[156] - b259[89]))

print("\nRETAINED GATE DATA")
for n259 in ("r243_raw", "r243_pre", "r243_active", "gate_ne186",
             "gate_bias186", "raw_base192", "active192", "test192",
             "gate_ne185", "coef_ne185", "coef192"):
    x259 = globals().get(n259)
    if x259 is None:
        continue
    if hasattr(x259, "shape"):
        print(n259, type(x259).__name__, x259.shape)
    else:
        print(n259, repr(x259)[:250])

for n259 in ("gated_ne185", "collect_channel186", "replay_factual125"):
    fn259 = globals()[n259]
    print(n259, "arguments",
          fn259.__code__.co_varnames[:fn259.__code__.co_argcount],
          "names", fn259.__code__.co_names,
          "constants", repr(fn259.__code__.co_consts)[:700])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 259\n"
            "Saved insulin innovation gain 1.5 / gate shift 28. Only "
            "z_post.0.weight/bias changed from r250_passing; backup r259_backup. "
            "CPU delta +.00374508. Stronger screened settings mostly exceed .005. "
            "GPU factual and dose rollout validation pending. Inspecting NE "
            "posterior rows 156/89 for an analogous selective calibration.\n")
```