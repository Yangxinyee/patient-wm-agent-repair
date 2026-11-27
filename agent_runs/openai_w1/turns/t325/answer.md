None of the tested contexts reproduces the posterior accurately. I won’t treat those counterfactual features as reliable. Instead, I’ll use the exported posterior to build factual prelude features and examine dose contrasts with that latent held fixed. Matching the verified original tokens will validate the factual construction; GPU tests will still be needed for any resulting edit.

```python
mu_actual285 = torch.from_numpy(S280["x"][:, 960:992])
slots285 = Hflow281 + W282["core.slots"][None]
raw285 = torch.cat((
    slots285, e277[:, None], u277[:, None],
    lin277(mu_actual285, "z_in", W282)[:, None]
), dim=1)
E285 = prelude277(raw285, W282).detach().numpy().copy()

# Independent factual-token check against the earlier verified original replay.
raworig285 = torch.cat((
    hflow277 + O277["core.slots"][None],
    e277[:, None], u277[:, None],
    lin277(torch.from_numpy(ds_s["x"][:, 960:992]),
           "z_in", O277)[:, None]
), dim=1)
rawcheck285 = raworig285[pick276] - rx_tokens126
print("Original factual-token check RMS:",
      float(torch.sqrt(torch.mean(rawcheck285**2))),
      "max:", float(rawcheck285.abs().max()))
print("Per-token RMS:",
      torch.sqrt(torch.mean(rawcheck285**2, dim=(0, 2))).tolist())
print("Actual current factual E:", E285.shape,
      "finite:", bool(np.isfinite(E285).all()))

Es285 = {}
for (drug, factor), uu in Ucf279.items():
    uu = torch.as_tensor(uu, dtype=torch.float32)
    rawcf = raw285.clone()
    rawcf[:, 5] = uu
    ecf = prelude277(rawcf, W282).detach().numpy().copy()
    Es285[(drug, factor)] = ecf
    j = 0 if drug == "insulin" else 1
    active = ds_rates[:, j] > 0
    contrast = ecf - E285
    print(drug, factor,
          "active contrast RMS",
          float(np.sqrt(np.mean(contrast[active]**2))),
          "inactive contrast RMS",
          float(np.sqrt(np.mean(contrast[~active]**2))))

print("\n=== SAFE CANDIDATE HISTORY ===")
text285 = Path("/work/notes.md").read_text()
for tag in ("## Run 235", "## Run 236", "## Run 237"):
    pos = text285.find(tag)
    if pos >= 0:
        end = text285.find("\n## ", pos + len(tag))
        if end < 0:
            end = len(text285)
        print(text285[pos:min(end, pos + 1500)])

print("\n=== CODA HELPER RETURN STRUCTURE ===")
zero285 = coda269(torch.zeros(192))
if isinstance(zero285, (tuple, list)):
    print(type(zero285).__name__,
          [tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
           for v in zero285])
else:
    print(type(zero285).__name__, getattr(zero285, "shape", None))
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 285\nPosterior-from-context reconstruction remains unvalidated. "
        "Built actual factual E using GPU-exported posterior, and ten provisional "
        "action-only dose contrasts with that posterior held fixed; independently "
        "checked original factual tokens against verified replay. No weights changed. "
        "Counts: 285 Python runs, 30 forwards.\n"
    )
```
