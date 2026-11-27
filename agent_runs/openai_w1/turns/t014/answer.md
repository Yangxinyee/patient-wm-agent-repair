The compact metadata is now available. I’ll rebuild the value head directly from the exported states and test a small set of plausible encodings, including prior versus posterior latent means. If one configuration closely matches the forward output, it will give us a reliable basis for an algebraic edit.

```python
import itertools
import torch.nn.functional as F
torch.set_num_threads(8)

cmp = aligned_ref.copy()
cmp["mean_error"] = head_test_predictions[best_key][0] - cmp.predicted_mean
cmp["nll_error"] = head_test_predictions[best_key][1] - cmp.nll
print("Existing reconstruction errors:")
print(cmp.groupby("event")[["mean_error", "nll_error"]].agg(["mean", "std"]).to_string())

rg = np.random.default_rng(712)
test_idx = np.sort(rg.choice(len(HM), min(800, len(HM)), replace=False))
ix = torch.as_tensor(test_idx)
hs = hx[ix].float()
dt = torch.as_tensor(HM[test_idx, 1], dtype=torch.float32)
hc = torch.as_tensor(HM[test_idx, 2], dtype=torch.long)
has = torch.as_tensor(HM[test_idx, 3], dtype=torch.float32)
age = torch.as_tensor(HM[test_idx, 4], dtype=torch.float32)
zv = torch.as_tensor(HM[test_idx, 5], dtype=torch.float32)
lb = torch.as_tensor(HM[test_idx, 6], dtype=torch.long)
tb = torch.as_tensor(HM[test_idx, 7], dtype=torch.long)
ct = hcentres[ix].float()
refm = torch.as_tensor(aligned_ref.predicted_mean.to_numpy()[test_idx])
refn = torch.as_tensor(aligned_ref.nll.to_numpy()[test_idx])
flowed = hs[:, 192:960].reshape(-1, 4, 192) * torch.exp(-dt[:, None, None] * W["core.flow.a"].exp()[None, None, :])
basic = hs[:, :192] + norm(flowed.mean(1), "core.out_norm")
fixed = W["embed.code.weight"][hc] + W["embed.bins.value.weight"][lb] + W["embed.bins.rate.weight"][0]
transition = W["embed_transition.value_bin.weight"][hc * 17 + lb]
grid_results = []
for ga, scale, pa in itertools.product(["silu", "gelu"], [0.1, 1.0], ["silu", "gelu"]):
    g = mlp(dt.clamp_min(1e-6).log()[:, None] * scale, "gap", ga)
    context = basic + g
    zp = mlp(context, "z_prior", pa)
    for latent in ["prior", "posterior", "zero"]:
        z = zp[:, :32] if latent == "prior" else hs[:, 960:] if latent == "posterior" else torch.zeros_like(zp[:, :32])
        view = context + lin(z, "z_head")
        for typ, flagcol, stat in itertools.product([0, 1], [3, 5, None], [0, 1]):
            num = torch.zeros((len(ix), 8))
            num[:, 0] = has
            num[:, 1] = torch.log1p(age) / 10
            num[:, 2] = zv
            if flagcol is not None:
                num[:, flagcol] = has
            wcode = norm(view + W["type_emb.weight"][typ], "query_norm") + fixed + lin(num, "last_numeric")
            wcode = wcode + W["last_status.weight"][(has.long() if stat else torch.zeros_like(lb))]
            for ha in ["silu", "gelu"]:
                lg = mlp(wcode, "attr_heads.value_bin", ha) + transition
                p = lg.softmax(-1)
                mm = (p * ct).sum(-1)
                nn = -p[torch.arange(len(ix)), tb].log()
                cfg = (ga, scale, pa, latent, typ, flagcol, stat, ha)
                grid_results.append((cfg, float(((mm-refm)**2).mean().sqrt()), 
                                     float(((nn-refn)**2).mean().sqrt()), float(nn.mean())))
grid_results.sort(key=lambda t: t[1])
print("\nBest new configurations:")
for row in grid_results[:12]:
    print(row)
print("\nPrior log-scale summary for best context:")
cfg = grid_results[0][0]
zp = mlp(basic + mlp(dt.log()[:, None] * cfg[1], "gap", cfg[0]), "z_prior", cfg[2])
print("mu:", np.quantile(zp[:, :32].numpy(), [0,.1,.5,.9,1]))
print("second half:", np.quantile(zp[:, 32:].numpy(), [0,.1,.5,.9,1]))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 10\nDirect reconstruction grid over gap scale, activation, latent source, query type, last-register flags and status. Still no edits.\n")
    f.write(str(grid_results[:5]) + "\nCounts: 10 Python runs, 3 forward passes.\n")
```
