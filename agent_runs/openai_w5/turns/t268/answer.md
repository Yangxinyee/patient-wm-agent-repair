The current channels are drug-specific, but insulin doubling remains too weak and both halving responses are much too small. I’ll test a modest strength increase using only the paired value-head columns. If both directions improve while factual NLL stays within budget, strength is part of the problem; worsening asymmetry would show that scaling alone is insufficient.

```python
from safetensors.torch import save_file
torch.set_num_threads(8)

checkpoint238 = {
    nm: torch.as_tensor(load(nm)).float().clone().contiguous()
    for nm in edited()
}
save_file(checkpoint238, "/work/checkpoint_run238.safetensors")
Cbase238 = torch.as_tensor(load("attr_heads.value_bin.2.weight")).float().clone()

def install_strength238(insulin_strength=1.0, ne_strength=1.0):
    c = Cbase238.clone()
    for sensor, pair in sensor_neurons.items():
        factor = insulin_strength if sensor.startswith("glucose") else ne_strength
        c[:, list(pair)] *= factor
    save_edit("attr_heads.value_bin.2.weight", c.contiguous())
    return c

Ctest238 = install_strength238(2.0, 1.2)
print("Installed relative strengths: insulin 2.0, norepinephrine 1.2")
print("Only changed tensor relative to saved candidate: attr_heads.value_bin.2.weight")

for k in [18, 19, 20]:
    p = f"/work/fwd_{k}/summary.json"
    if os.path.isfile(p):
        j = json.load(open(p))
        print("forward", k, "summary:", str(j.get("results", j))[:1900])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 238: GPU-calibrated strength trial\n"
            "- Saved complete pre-trial edits in /work/checkpoint_run238.safetensors.\n"
            "- Multiplied paired glucose value-head columns by 2.0 and MAP columns by 1.2; all transport and centering weights unchanged.\n"
            "- This is a calibration trial, not an accepted repair. Requires paired GPU dose rollouts and factual NLL validation.\n"
            "- Scaling alone is refuted if doubling/halving asymmetry persists or factual value-NLL cost exceeds +0.005.\n")
```
