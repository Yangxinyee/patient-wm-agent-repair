All 4,510 deterministic predictions match the GPU closely. The record-based estimates remain about −1.99 glucose units per insulin rate unit and +3.29 MAP units per norepinephrine rate unit. I’ll check the latent prior next. Its simulated sampling variability should explain the original factual outputs; excess error would mean the stochastic reconstruction is still wrong.

```python
import time
import numpy as np
import torch
from pathlib import Path

print("=== ALIGNED INNOVATION TARGETS ===")
for drug98 in ["insulin", "norepinephrine"]:
    a98 = repair_aligned[drug98]
    print(drug98, list(a98.columns))
    print(a98.head(3).to_string(index=False))
    print("DML cache keys:", list(dml_cache[drug98]))
    print("causal estimate:", dml_results[drug98]["all active"])

ctx98 = base97 - W96["z_head.bias"]
scale98 = torch.tensor([
    r90_numeric[(e, "value")]["iqr"] for e in meta96["event"]
], dtype=torch.float32)
original_mean98 = torch.tensor(meta96["tool_mean"].to_numpy(), dtype=torch.float32)
original_nll98 = torch.tensor(meta96["tool_nll"].to_numpy(), dtype=torch.float32)

gen98 = torch.Generator().manual_seed(9801)
eps98_half = torch.randn((16, len(meta96), 32), generator=gen98)
eps98 = torch.cat([eps98_half, -eps98_half], dim=0)
results98 = []
cache98 = {}

for pa98 in ["gelu", "silu"]:
    pp98 = mlp96(ctx98, "z_prior", pa98)
    for packing98 in ["split", "interleaved"]:
        if packing98 == "split":
            pm98, ls98 = pp98[:, :32], pp98[:, 32:]
        else:
            pm98, ls98 = pp98[:, 0::2], pp98[:, 1::2]
        ls98 = ls98.clamp(-6, 3)
        zz98 = pm98[None] + eps98 * ls98.exp()[None]
        latent98 = torch.nn.functional.linear(
            zz98, W96["z_head.weight"], W96["z_head.bias"])
        view98 = ctx98[None] + latent98
        qq98 = norm96(view98 + W96["type_emb.weight"][1], "query_norm")
        xx98 = qq98 + fixed96[None]
        logits98 = mlp96(xx98, "attr_heads.value_bin", "gelu") + trans96[None]
        probs98 = logits98.softmax(-1)
        drawmean98 = (probs98 * centres96[None]).sum(-1)
        mcmean98 = drawmean98.mean(0)
        mcnll98 = -probs98.mean(0).gather(1, obin96[:, None]).squeeze(1).log()
        mse98 = float(((mcmean98 - original_mean98) / scale98).square().mean())
        expected98 = float((drawmean98.var(0) / (4 * scale98.square())).mean())
        result98 = {
            "activation": pa98, "packing": packing98,
            "normalized_MSE": mse98,
            "expected_four_draw_MSE": expected98,
            "excess_MSE": mse98 - expected98,
            "NLL_RMSE": float((mcnll98 - original_nll98).square().mean().sqrt()),
            "mean_log_sd": float(ls98.mean()),
        }
        results98.append(result98)
        print(result98)
        if pa98 == "gelu" and packing98 == "split":
            cache98 = {"mean": pm98, "log_sd": ls98, "q": qq98,
                       "with_code": xx98, "probs": probs98,
                       "predicted_mean": mcmean98, "nll": mcnll98}

print("CURRENT EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 98: stochastic head reconstruction\n"
            "Testing latent prior activation/packing against original factual "
            "outputs, allowing for the GPU's four prior draws. "
            "Recovered aligned innovation targets. No weights edited.\n")
    for result98 in results98:
        f.write(repr(result98) + "\n")
```
