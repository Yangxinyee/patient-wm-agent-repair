The records support nonzero effects: insulin lowers subsequent glucose, and norepinephrine raises subsequent MAP. The estimated rollout targets remain assumption-dependent; they are not ground truth. I’ll test shorter insulin persistence, with a corresponding partial rollback of its factual-centering correction. Success requires passing the NLL limit while retaining better dose responses than the earlier feasible edit.

```python
import torch, pandas as pd, inspect
from safetensors.torch import load_file as st_load, save_file as st_save

st_save(
    {n: torch.as_tensor(load(n)).contiguous() for n in edited()},
    "/work/checkpoint_run285.safetensors"
)
cp282_285 = st_load("/work/checkpoint_run282.safetensors")
cp264_285 = st_load("/work/checkpoint_run264.safetensors")

# Undo the unsuccessful temperature trial.
for name in (
    "attr_heads.value_bin.2.weight",
    "attr_heads.value_bin.2.bias",
    "embed_transition.value_bin.weight",
):
    if name in cp282_285:
        save_edit(name, cp282_285[name])
    else:
        drop_edit(name)

# Shorter insulin persistence; keep NE and other transport unchanged.
flow285 = torch.as_tensor(load("core.flow.a")).clone()
flow285[[21, 61]] = -np.log(900.0)
save_edit("core.flow.a", flow285)

# Interpolate only the incremental centering correction, not drug coefficients.
for name in ("attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias"):
    before = cp264_285[name]
    after = cp282_285[name]
    save_edit(name, before + 0.25 * (after - before))

print("Insulin timescales:", torch.exp(-flow285[[21, 61]]).tolist())
print("Strength helper:", inspect.signature(install_strength238))
print("Helper constants:", repr(install_strength238.__code__.co_consts)[:1000])
print("Helper dependencies:", install_strength238.__code__.co_names)

n285 = open("/work/notes.md").read()
for marker in ("## Run 140", "## Runs 140", "## Run 142", "## Run 150"):
    pos = n285.find(marker)
    if pos >= 0:
        print("HEAD DESIGN:", n285[pos:pos+2600])
        break

tuning_ids285 = pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].drop_duplicates().tolist()
print("Tuning A IDs:", tuning_ids285[:100])
print("Reserved head rows:",
      torch.nonzero(torch.as_tensor(load("attr_heads.value_bin.0.weight"))[:, [21,61,15,66]]
                    .abs().max(dim=1).values > 1e-5).flatten().tolist())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 285: reversible shorter-persistence trial\n"
            "- Saved the failed tau=1500/temperature candidate in checkpoint_run285.safetensors.\n"
            "- Undid the 0.97 temperature scaling. Set insulin private coordinates 21/61 to tau=900 seconds.\n"
            "- Retained 25% of the incremental tau=1500 factual-centering correction; baseline centers and drug coefficients unchanged.\n"
            "- NE and other transport unchanged. This is unaccepted pending paired GPU factual and rollout checks.\n")
```
