The evaluator returns per-reading arrays; the displayed rows were mostly inactive and did not reveal the scaling cost. I’ll now summarize NLL changes over all readings and held-out stays, and verify the bias compensation. This will identify whether a stronger candidate has enough factual margin for GPU validation.

```python
import numpy as np
import torch

r111_results = {}
print("strength | delta NLL all/train/held-out | active-glucose mean-change RMS")
for strength111 in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
    nll111, means111 = local104(strength111)
    delta111 = nll111 - baseNLL104
    base_means111 = np.sum(Pmean104 * C104, axis=1)
    mean_change111 = means111 - base_means111
    stats111 = (
        float(delta111.mean()),
        float(delta111[trainmask99].mean()),
        float(delta111[valmask99].mean()),
        float(np.sqrt(np.mean(mean_change111[np.asarray(active104, dtype=bool)]**2)))
    )
    r111_results[strength111] = stats111
    print(strength111, tuple(round(x, 7) for x in stats111))

bias_expected111 = (
    torch.as_tensor(b2103).double()
    + torch.as_tensor(W2103).double()[:, pair104]
      @ torch.as_tensor(hbar104).double()
)
print("Bias compensation max error:",
      float((r107_backup["attr_heads.value_bin.2.bias"].double()
             - bias_expected111).abs().max()))
print("Installed layer-2 pair columns versus symmetric tilt construction:")
for position111, neuron111 in enumerate(pair104):
    coef111 = r107_backup["attr_heads.value_bin.2.weight"][:, neuron111].numpy()
    print(int(neuron111),
          "tilt coefficient", float(np.dot(coef111, tilt104) / np.dot(tilt104, tilt104)),
          "orthogonal residual RMS",
          float(np.sqrt(np.mean((coef111 -
                np.dot(coef111, tilt104) / np.dot(tilt104, tilt104) * tilt104)**2))))
print("Original pair layer-2 RMS:",
      float(torch.as_tensor(W2103)[:, pair104].square().mean().sqrt()))
```