The largest non-initial errors occur in measurement bundles, while isolated drug starts match closely. I’ll test measurement unit and quality embeddings against the saved posterior states. A near-zero error would identify the encoding convention; no close match would mean another component is missing.

```python
import torch, itertools
torch.set_num_threads(8)
print("latent tensors:", [(n, SHAPES[n]) for n in NAMES if "z_" in n or "post" in n])
print("atom helper constants:", recon_atom.__code__.co_consts)
print("MLP constants:", recon_mlp.__code__.co_consts)
POST_PREFIX = next(n[:-9] for n in NAMES if n.endswith(".0.weight") and "post" in n)

def single_encode_batch(atoms, prefix="encode"):
    n = len(atoms)
    size = torch.full((n, 1), float(np.log(2)), dtype=torch.float32)
    mid = recon_mlp(atoms, prefix + ".mlp1") + recon_lin(size, prefix + ".size")
    return recon_norm(recon_mlp(mid, prefix + ".mlp2"), prefix + ".norm")

measurement_unit_search = {}
for code in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    ii = []
    for i in recon_valid_rows:
        if len(recon_obs_lists[i]) == 1 and not recon_action_lists[i]:
            row = state_df.iloc[i]
            sub = tune[(tune.stay_id == row.stay_id) & (np.abs(tune.hour-row.hour) < 1e-5)]
            if len(sub) == 1 and sub.iloc[0].event == code:
                ii.append(int(i))
        if len(ii) >= 60:
            break
    if not ii:
        continue
    atoms = torch.stack([recon_obs_lists[i][0] for i in ii])
    choices0 = list(itertools.product(range(6), range(4), range(4)))
    deltas = torch.stack([
        W0["embed.categorical.unit.weight"][u] -
        W0["embed.categorical.unit.weight"][RECON_UNITS[code]] +
        W0["embed.categorical.rate_unit.weight"][r] -
        W0["embed.categorical.rate_unit.weight"][0] +
        W0["embed.kq.weight"][k] - W0["embed.kq.weight"][0]
        for u, r, k in choices0
    ])
    aa = (atoms[None] + deltas[:, None]).reshape(-1, 192)
    ee0 = single_encode_batch(aa)
    cc0 = recon_context[ii].repeat(len(choices0), 1)
    pred0 = recon_mlp(torch.cat([cc0, ee0, torch.zeros_like(ee0)], dim=1),
                      POST_PREFIX)[:, :32].reshape(len(choices0), len(ii), 32)
    target0 = torch.as_tensor(S0["x"][ii, 960:992])
    losses0 = ((pred0-target0[None])**2).mean(dim=(1, 2)).sqrt().numpy()
    order0 = np.argsort(losses0)[:3]
    measurement_unit_search[code] = [(choices0[j], float(losses0[j])) for j in order0]
    print(code, "n", len(ii), "best:", measurement_unit_search[code])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 23: measurement embedding search\n"
            + str(measurement_unit_search) +
            "\nNo weight edits. Python runs: 23; forward passes: 3.\n")
```
