The probe identifies the reading register as presence in field 0, log-age in seconds divided by 10 in field 3, and normalized last value in field 6, with left-sided bins and status zero. I’ll rebuild the deterministic head using those fields. Agreement with the saved GPU means and NLLs would validate it; substantial disagreement would refute the reconstruction.

```python
import itertools, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

torch.set_num_threads(8)
torch.set_grad_enabled(False)
t96 = time.time()
W96 = {n: torch.as_tensor(load_original(n), dtype=torch.float32) for n in NAMES}

def lin96(x, p):
    return F.linear(x, W96[p + ".weight"], W96[p + ".bias"])

def act96(x, kind):
    return {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}[kind](x)

def mlp96(x, p, kind):
    return lin96(act96(lin96(x, p + ".0"), kind), p + ".2")

def norm96(x, p):
    return F.layer_norm(x, (192,), W96[p + ".weight"], W96[p + ".bias"])

meta96 = r90_meta.copy()
tool96 = pd.read_csv("/work/fwd_8/factual_edited.csv")
keys96 = ["stay_id", "hour", "event"]
aligned96 = meta96[keys96].merge(tool96, on=keys96, how="left",
                               validate="one_to_one", sort=False)
assert len(aligned96) == len(meta96) and aligned96["predicted_mean"].notna().all()
tm96 = torch.tensor(aligned96["predicted_mean"].to_numpy(), dtype=torch.float32)
tn96 = torch.tensor(aligned96["nll"].to_numpy(), dtype=torch.float32)

code96 = torch.as_tensor(r90_codeid, dtype=torch.long)
lb96 = torch.as_tensor(r90_bins["left"], dtype=torch.long)
num96 = torch.zeros((len(meta96), 8))
num96[:, 0] = torch.tensor(meta96["has_last"].to_numpy(), dtype=torch.float32)
num96[:, 3] = torch.tensor(r90_age_options["state_seconds"], dtype=torch.float32)
num96[:, 6] = torch.tensor(meta96["last_z"].to_numpy(), dtype=torch.float32)
reg96 = (lin96(num96, "last_numeric")
         + W96["embed.bins.value.weight"][lb96]
         + W96["embed.bins.rate.weight"][0]
         + W96["last_status.weight"][0])
fixed96 = reg96 + W96["embed.code.weight"][code96]
trans96 = W96["embed_transition.value_bin.weight"][code96 * 17 + lb96]
centres96 = torch.as_tensor(r90_centres, dtype=torch.float32)
obin96 = torch.as_tensor(r91_obin, dtype=torch.long)
out96 = torch.as_tensor(wr_out, dtype=torch.float32)
h96 = torch.as_tensor(wr_h, dtype=torch.float32)
dt96 = torch.as_tensor(wr_dt, dtype=torch.float32).reshape(-1, 1)
print("Aligned readings:", len(meta96), "type embedding:", W96["type_emb.weight"].shape)

ix96 = torch.linspace(0, len(meta96) - 1, 1000).long().unique()
gap_inputs96 = {
    "log1p_seconds/10": torch.log1p(dt96) / 10,
    "log_seconds/10": torch.log(dt96.clamp_min(1e-6)) / 10,
    "log1p_seconds": torch.log1p(dt96),
    "log_seconds": torch.log(dt96.clamp_min(1e-6)),
    "log1p_hours/10": torch.log1p(dt96 / 3600) / 10,
    "log1p_hours": torch.log1p(dt96 / 3600),
}
bases96 = {}
for flow96, unit96 in [("seconds", 1.), ("hours", 3600.), ("none", None)]:
    hh96 = h96 if unit96 is None else h96 * torch.exp(
        -(dt96 / unit96).reshape(-1, 1, 1) * W96["core.flow.a"].exp())
    bases96[flow96] = out96 + norm96(hh96.mean(1), "core.out_norm") + W96["z_head.bias"]

results96 = []
for flow96, gap96, ga96, ha96, type96 in itertools.product(
        bases96, gap_inputs96, ["gelu", "silu", "relu"],
        ["gelu", "silu"], range(W96["type_emb.weight"].shape[0])):
    view96 = bases96[flow96][ix96] + mlp96(gap_inputs96[gap96][ix96], "gap", ga96)
    q96 = norm96(view96 + W96["type_emb.weight"][type96], "query_norm")
    logits96 = mlp96(q96 + fixed96[ix96], "attr_heads.value_bin", ha96) + trans96[ix96]
    probs96 = logits96.softmax(-1)
    mean96 = (probs96 * centres96[ix96]).sum(-1)
    nll96 = -logits96.log_softmax(-1).gather(1, obin96[ix96, None]).squeeze(1)
    scale96 = torch.tensor([
        r90_numeric[(e, "value")]["iqr"] for e in meta96["event"]
    ], dtype=torch.float32)[ix96]
    score96 = float(torch.sqrt(((mean96 - tm96[ix96]) / scale96).square().mean()))
    results96.append((score96, float(torch.sqrt((nll96 - tn96[ix96]).square().mean())),
                      flow96, gap96, ga96, ha96, type96))
results96.sort()
print("BEST DETERMINISTIC CONFIGURATIONS:")
for result96 in results96[:8]:
    print(result96)
print("Elapsed seconds:", round(time.time() - t96, 1))
print("CURRENT EDITS:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 96: probe-resolved reading register\n"
        "Controlled probe identifies numeric fields 0=presence, "
        "3=log1p(age_seconds)/10, 6=last normalized value; other reading "
        "fields zero, left-sided bins, measurement status zero. "
        "Testing deterministic reconstruction against forward 8. No edits.\n"
    )
```
