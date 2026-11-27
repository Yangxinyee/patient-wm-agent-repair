The recovered record-based targets are roughly −13.6/+6.8 for insulin and +4.2–4.6/−2.1–2.3 for norepinephrine. I’ll measure reading-logit sensitivities at the exact GPU anchor and locate the corrected dose contrasts. These will test whether different hidden directions can preserve drug-response changes while reducing factual leakage.

```python
import numpy as np, torch, torch.nn.functional as F
import pandas as pd

defaults136 = readout121.__defaults__
print("Reading defaults:", [(tuple(x.shape), str(x.dtype)) if hasattr(x, "shape")
                            else type(x).__name__ for x in defaults136])
mean_base136, logits_base136, prob_base136, hidden_base136 = readout125(
    gpu_h134, gpu_out134)
print("Anchored readout matches previous GPU-state decoder:",
      float(np.sqrt(np.mean((mean_base136.numpy() - mstate125) ** 2))))

basis136 = torch.as_tensor(Dout120, dtype=torch.float32)
prebasis136 = F.linear(basis136.T, W0["core.c_out.weight"], None).T
eps136 = 0.1
Jmean136 = np.empty((len(mean_base136), basis136.shape[1]), np.float32)
Jlogit136 = np.empty((len(mean_base136), 17, basis136.shape[1]), np.float32)
for j136 in range(basis136.shape[1]):
    plus_h136 = torch.empty_like(gpu_h134)
    minus_h136 = torch.empty_like(gpu_h134)
    plus_o136 = torch.empty_like(gpu_out134)
    minus_o136 = torch.empty_like(gpu_out134)
    shift136 = eps136 * prebasis136[:, j136][None, None, :]
    for start136 in range(0, len(precoda134), 256):
        sl136 = slice(start136, start136 + 256)
        plus_h136[sl136], plus_o136[sl136] = coda_input134(
            precoda134[sl136] + shift136)
        minus_h136[sl136], minus_o136[sl136] = coda_input134(
            precoda134[sl136] - shift136)
    mp136, lp136, _, _ = readout125(plus_h136, plus_o136)
    mm136, lm136, _, _ = readout125(minus_h136, minus_o136)
    Jmean136[:, j136] = ((mp136-mm136)/(2*eps136)).numpy()
    Jlogit136[:, :, j136] = ((lp136-lm136)/(2*eps136)).numpy()
print("Mean-sensitivity RMS by basis direction:",
      np.round(np.sqrt(np.mean(Jmean136**2, axis=0)), 4))

print("\nCorrected contrast arrays:")
for n136, v136 in list(globals().items()):
    if isinstance(v136, (np.ndarray, torch.Tensor)) and (
        v136.ndim >= 2 and v136.shape[0] in [1497, 5988]
        and (v136.shape[-1] in [192, 188, 32] or n136.endswith(("93", "94", "95")))
    ):
        print(n136, tuple(v136.shape))
print("\nContrast containers/metadata:")
for n136, v136 in list(globals().items()):
    if isinstance(v136, dict):
        matching136 = [(k, tuple(x.shape)) for k, x in v136.items()
                       if hasattr(x, "shape") and len(x.shape) >= 1
                       and x.shape[0] in [1497, 5988]]
        if matching136:
            print(n136, matching136[:6])
    elif isinstance(v136, pd.DataFrame) and len(v136) in [1497, 5988, 4356, 4751]:
        print(n136, tuple(v136.shape), list(v136.columns)[:12])
    elif isinstance(v136, (list, tuple)) and len(v136) == 1497:
        print(n136, type(v136).__name__, "first", repr(v136[0])[:250])

with open("/work/notes.md", "a") as f136:
    f136.write("\n## Run 136: anchored distribution sensitivities\n"
               "Measured central finite-difference mean and fine-logit sensitivities "
               "for 14 hnew directions using exact pre-coda GPU anchors. Recovering "
               "corrected dose contrasts for matrix regression with factual Fisher "
               "constraints. No weights changed. Python runs: 136; forwards: 15.\n")
```