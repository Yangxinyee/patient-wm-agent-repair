The posterior signal is strongly attenuated before reaching the output, and the full CPU replay is still too inaccurate for reliable rollout testing. I’ll check a localized alternative: use two latent coordinates to carry dose residuals into the state directly. This is viable only if the residuals are small on factual records and respond proportionally to edited doses.

```python
import numpy as np
import torch

print("LATENT INPUT TENSORS")
for n186 in NAMES:
    if "z_in" in n186:
        a186 = np185(load_original(n186))
        print(n186, a186.shape, "rms", float(np.sqrt(np.mean(a186**2))),
              "range", [float(a186.min()), float(a186.max())])

print("\nBRANCH SPECIFICATIONS")
print("rp_branch_specs keys", list(rp_branch_specs))
for drug186, spec186 in rp_branch_specs.items():
    print(drug186)
    for key186, val186 in spec186.items():
        if hasattr(val186, "shape"):
            a186 = np185(val186)
            print(key186, a186.shape, "rms", float(np.sqrt(np.mean(a186**2))))
        else:
            print(key186, str(val186)[:200])

print("\nEXISTING HEAD ACCURACY / CONTRAST")
for key186 in ["he_rmse", "he_nll_diff", "hp_sensitivity"]:
    if key186 in globals():
        val186 = globals()[key186]
        if hasattr(val186, "shape"):
            print(key186, np185(val186).shape, np185(val186).reshape(-1)[:12])
        else:
            print(key186, val186)

print("\nPOSTERIOR BRANCH FACTUAL AND EDITED ACTIVATIONS")
x186 = torch.tensor(np185(rp_input), dtype=torch.float32)
for drug186, spec186 in rp_branch_specs.items():
    gw186 = torch.tensor(np185(spec186["gw"]), dtype=torch.float32)
    rw186 = torch.tensor(np185(spec186["rw"]), dtype=torch.float32)
    gb186, rb186 = float(spec186["gb"]), float(spec186["rb"])
    def branch186(xx186):
        gg186 = xx186 @ gw186 + gb186
        rr186 = xx186 @ rw186 + rb186
        return torch.nn.functional.gelu(gg186 + rr186) - torch.nn.functional.gelu(gg186)
    bf186 = branch186(x186).numpy()
    print(drug186, "factual RMS", float(np.sqrt(np.mean(bf186**2))),
          "quantiles", np.quantile(bf186, [0,.01,.5,.99,1]).round(5).tolist())
    for factor186 in [0., .5, 2., 3.]:
        kk186 = (drug186, factor186)
        if kk186 in rp_cf_input:
            bc186 = branch186(torch.tensor(np185(rp_cf_input[kk186]), dtype=torch.float32)).numpy()
            mm186 = np.asarray(rp_cf_masks[drug186], dtype=bool)
            print("factor", factor186, "mean contrast",
                  float(np.mean((bc186-bf186)[mm186])),
                  "RMS contrast", float(np.sqrt(np.mean((bc186-bf186)[mm186]**2))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 186\nPosterior-to-h gain is weak; full CPU core replay error remains too large. "
            "Checking whether a localized two-coordinate dose-residual carrier can drive z_in directly, "
            "instead of injecting into observational latent directions. No edits.\n")
```