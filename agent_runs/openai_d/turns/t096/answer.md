The deterministic scan reduced the discrepancy substantially, but it is still far above rounding error. The register layout therefore remains unverified. Rather than keep guessing, I’ll install a temporary linear probe that removes the query contribution and makes the numeric register directly measurable. The probe will confirm the layout only if simple record-derived features explain its output to numerical precision.

```python
import numpy as np
import torch
from pathlib import Path

# Diagnostic only: encode an identifiable projection of all eight register inputs.
P86 = {}
P86["coefficients"] = torch.tensor(
    [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.float32
) / 128.0

L86 = load_original("last_numeric.weight").double()
P86["projection"] = torch.linalg.lstsq(
    L86.T, P86["coefficients"].double()[:, None]
).solution[:, 0].float()
print("Projection identity error:",
      float((L86.T @ P86["projection"].double()
             - P86["coefficients"].double()).abs().max()))
print("Projection norm:", float(P86["projection"].norm()))

pW86 = torch.zeros_like(load_original("attr_heads.value_bin.0.weight"))
pb86 = torch.zeros_like(load_original("attr_heads.value_bin.0.bias"))
pW86[0] = P86["projection"]
pb86[0] = 30.0

pO86 = torch.zeros_like(load_original("attr_heads.value_bin.2.weight"))
po86 = torch.zeros_like(load_original("attr_heads.value_bin.2.bias"))
P86["bin_slopes"] = torch.linspace(-1.5, 1.5, 16)
pO86[1:, 0] = P86["bin_slopes"]
po86[1:] = -30.0 * P86["bin_slopes"]
po86[0] = -50.0

save_edit("query_norm.weight", torch.zeros_like(load_original("query_norm.weight")))
save_edit("query_norm.bias", torch.zeros_like(load_original("query_norm.bias")))
save_edit("embed_transition.value_bin.weight",
          torch.zeros_like(load_original("embed_transition.value_bin.weight")))
save_edit("attr_heads.value_bin.0.weight", pW86)
save_edit("attr_heads.value_bin.0.bias", pb86)
save_edit("attr_heads.value_bin.2.weight", pO86)
save_edit("attr_heads.value_bin.2.bias", po86)

print("Register projection coefficients:", P86["coefficients"].tolist())
print("Temporary diagnostic edits:", edited())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Runs 85–86: identifiable register probe\n"
        "Deterministic head scan did not achieve exact agreement: best normalized "
        "mean RMSE 0.02094 and NLL RMSE 0.10588. Temporarily zeroed query_norm "
        "and value transitions, and installed an analytically constructed linear "
        "value-head probe. Its scalar register coefficients are powers of two /128, "
        "so observed probe means can identify the actual eight-input register layout. "
        "These edits are diagnostics only and must all be reverted.\n"
    )
```