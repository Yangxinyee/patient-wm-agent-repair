The replay inputs and exported targets are available, so I can test individual stages without another forward pass. I’ll check attention reshaping, normalization, and input alignment. An incorrect head count or stale input could explain the persistent error; if those checks pass, I’ll focus on the recurrent mixing equation.

```python
import dis, numpy as np, torch

print("=== CONFIGURATION ===")
print(VOCAB["config"])

print("\n=== INPUT ALIGNMENT AND SCALE ===")
for name in ("rx_hflow", "rx_e", "rx_u", "rx_z", "rx_targeth", "rx_targetout",
             "rx_E126", "rx_delta126", "rx_abar126", "rx_dt"):
    x = globals()[name].detach().cpu().numpy()
    print(name, "shape", x.shape, "mean", float(x.mean()),
          "std", float(x.std()), "range", (float(x.min()), float(x.max())))

_ix130 = np.asarray(rx_pick)
for label, a, b in (
    ("e versus rr_e4", rx_e, torch.as_tensor(rr_e4[_ix130])),
    ("u versus repaired empty-set encoding", rx_u, torch.as_tensor(rr_base_u112[_ix130])),
    ("z versus exported posterior", rx_z, torch.as_tensor(rr_zexport4[_ix130])),
):
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    print(label, "RMSE", float((a-b).square().mean().sqrt()))

print("\n=== ATTENTION AND NORMALIZATION IMPLEMENTATIONS ===")
for name, limit in (("rx_norm", 1000), ("rx_attn", 2100),
                    ("rx_layer2", 1800), ("rx_cross126", 2400)):
    obj = globals()[name]
    print("\n", name)
    print("constants:", repr(obj.__code__.co_consts)[:450])
    print("names:", obj.__code__.co_names)
    print(" | ".join(i.opname + " " + i.argrepr
                     for i in dis.get_instructions(obj))[:limit])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n\n## Run 130\n"
        "Checking attention layout, model configuration, and alignment of replay "
        "inputs with exact encodings and exported posteriors. No weights edited. "
        "Counts: 130 Python runs, 6 forward passes.\n"
    )
```