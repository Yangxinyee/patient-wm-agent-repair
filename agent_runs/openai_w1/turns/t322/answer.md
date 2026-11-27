All bundle identities and hours align, and the current model’s incoming state differs modestly from the original’s. I’ll now reconstruct its posterior and dose-edited prelude features. Agreement with the exported posterior is the next check: a mismatch would invalidate any projection fitted from these features.

```python
import torch.nn.functional as F
torch.set_grad_enabled(False)
W282 = {n: torch.as_tensor(load(n), dtype=torch.float32).clone() for n in NAMES}

# Anchor to the verified original context, preserving gap and initial-state terms.
delta_out282 = torch.zeros_like(ctx277)
delta_out282[ix281] = (
    Oprev281[ix281]
    - torch.as_tensor(ds_s["x"][ix281 - 1, :192], dtype=torch.float32)
)
ctx282 = (
    ctx277 + delta_out282
    + norm277(Hflow281.mean(1), "core.out_norm", W282)
    - norm277(hflow277.mean(1), "core.out_norm", W282)
)
inp282 = torch.cat((ctx282, e277, u277), dim=-1)
mu282 = mlp277(inp282, "z_post", W282)[:, :32]
zerr282 = mu282 - torch.from_numpy(S280["x"][:, 960:992])
zrms282 = float(torch.sqrt(torch.mean(zerr282**2)))
zmax282 = float(zerr282.abs().max())
print("Current posterior reconstruction: RMS error", zrms282,
      "maximum error", zmax282)
print("Original prelude helper repeat RMS:",
      float(torch.sqrt(torch.mean(
          (prelude277(raw277, W277) - torch.as_tensor(Ef277))**2
      ))))

Es282 = {}
if zrms282 < 0.002 and zmax282 < 0.02:
    slots282 = Hflow281 + W282["core.slots"][None]
    raw282 = torch.cat((
        slots282, e277[:, None], u277[:, None],
        lin277(mu282, "z_in", W282)[:, None]
    ), dim=1)
    E282 = prelude277(raw282, W282).detach().numpy().copy()
    print("Current factual prelude:", E282.shape,
          "; RMS change from corrected original-state features:",
          float(np.sqrt(np.mean((E282 - Ef277)**2))))

    for (drug, factor), uu in Ucf279.items():
        uu = torch.as_tensor(uu, dtype=torch.float32)
        mucf = mlp277(torch.cat((ctx282, e277, uu), -1),
                      "z_post", W282)[:, :32]
        rawcf = torch.cat((
            slots282, e277[:, None], uu[:, None],
            lin277(mucf, "z_in", W282)[:, None]
        ), dim=1)
        ecf = prelude277(rawcf, W282).detach().numpy().copy()
        Es282[(drug, factor)] = ecf
        j = 0 if drug == "insulin" else 1
        active = ds_rates[:, j] > 0
        print(drug, factor,
              "active bundles", int(active.sum()),
              "active contrast RMS",
              float(np.sqrt(np.mean((ecf[active] - E282[active])**2))),
              "inactive contrast RMS",
              float(np.sqrt(np.mean((ecf[~active] - E282[~active])**2))))
else:
    print("RECONSTRUCTION FAILED: no projection will be fitted from these features.")

print("\n=== RECURRENT EDIT INTERFACE ===")
for n in ("core.b_in.weight", "core.b_in.bias", "core.delta_in",
          "core.c_out.weight", "core.c_out.bias"):
    print(n, SHAPES.get(n, "absent"))
for n in ("B257", "B271", "P271"):
    if n in globals():
        v = np276(globals()[n])
        print(n, v.shape, "column RMS", np.sqrt(np.mean(v*v, axis=0)))
for n in ("direct_readout257", "coda269", "rx_layer2",
          "rx_memory_for", "posterior_token270"):
    fn = globals().get(n)
    if inspect.isfunction(fn):
        c = fn.__code__
        print(n, "args", c.co_varnames[:c.co_argcount],
              "uses", c.co_names,
              "constants", [x for x in c.co_consts
                            if isinstance(x, (int, float, str))][:18])

with open("/work/notes.md", "a") as f:
    f.write(
        f"\n## Run 282\nRebuilt current posterior from aligned exported previous states; "
        f"posterior RMS/max error {zrms282:.6g}/{zmax282:.6g}. "
        f"Built {len(Es282)} corrected counterfactual prelude feature sets if the "
        f"reconstruction passed. No weights changed. Counts: 282 Python runs, 30 forwards.\n"
    )
```
