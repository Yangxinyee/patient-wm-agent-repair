The local decoder uses the prior latent mean, whereas the GPU averages sampled probabilities, so their absolute NLLs may differ. Before adding latent sampling, I’ll check the fixed decoder inputs against the existing reconstruction. Exact agreement would support the register mapping; any systematic discrepancy must be resolved before calibration.

```python
import torch, numpy as np, re
from pathlib import Path

torch.set_grad_enabled(False)
tune_meta167 = reading_metadata162(tune_records162, tune_archive162, "left")

def build_static167(meta):
    reg = np.zeros((len(meta), 8), dtype=np.float32)
    reg[:, 0] = meta.has_last.to_numpy()
    reg[:, 3] = np.log1p(meta.age.to_numpy()) / 10.0
    reg[:, 6] = meta.last_z.to_numpy()
    reg = torch.tensor(reg)
    code = torch.tensor(meta.code.to_numpy(), dtype=torch.long)
    lastbin = torch.tensor(meta.last_bin.to_numpy(), dtype=torch.long)
    static = (
        W0["embed.code.weight"][code]
        + linear0(reg, "last_numeric")
        + W0["embed.bins.value.weight"][lastbin]
        + W0["embed.bins.rate.weight"][0]
        + W0["last_status.weight"][0]
    )
    transition = W0["embed_transition.value_bin.weight"][code * 17 + lastbin]
    return static, transition, reg

static167, transition167, register167 = build_static167(tune_meta167)
print("STATIC COMPARISONS")
for k in ["head_static", "static_all122", "base162"]:
    other = globals()[k]
    diff = static167 - other
    print(k, "max", float(diff.abs().max()), "rms", float(diff.square().mean().sqrt()))
    for ev in tune_meta167.event.unique():
        mask = torch.tensor(tune_meta167.event.eq(ev).to_numpy())
        d = diff[mask]
        print(" ", ev, "rms", float(d.square().mean().sqrt()),
              "within-group residual rms", float((d - d.mean(0)).square().mean().sqrt()))

print("TRANSITION MAX ERROR", float((transition167 - head_transition).abs().max()))
print("METADATA ORDER", len(tune_meta167), len(head_meta),
      np.max(np.abs(tune_meta167.hour.to_numpy() - head_meta.hour.to_numpy())))
print("FUNCTION fun160")
print(compact_bytecode166(fun160, 1600))

p167 = r160[2].detach().cpu().numpy()
b167 = tune_meta167.target_bin.to_numpy(dtype=int)
nll167 = -np.log(np.maximum(p167[np.arange(len(b167)), b167], 1e-30))
print("DETERMINISTIC LOCAL NLL", nll167.mean())
print("GPU ORIGINAL NLL", orig159.nll.mean())
print("LOCAL/GPU MEAN RMSE",
      np.sqrt(np.mean((r160[0].numpy() - orig159.predicted_mean.to_numpy())**2)))
print("CENTRES EXAMPLE", repr(VOCAB["value_bin_centres"])[:650])

sections167 = re.split(r"\n(?=## )", Path("/work/notes.md").read_text())
print("\nRECONSTRUCTION NOTES")
for s in sections167:
    if re.search(r"(Run 11\b|Run 12\b|Run 121\b|Run 122\b|Run 125\b)", s.splitlines()[0] if s else ""):
        print(s[:1000])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 167: decoder static-input audit\n"
        "Built fresh static inputs with register columns 0=has_last, 3=log1p(age)/10, "
        "6=last_z and left-edge bins. Compared against retained decoder inputs and "
        "checked deterministic local/GPU NLL and means. No weights changed. "
        "Python runs: 167; forwards: 22.\n"
    )
```