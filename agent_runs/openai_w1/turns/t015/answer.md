The prior’s latent variability is substantial, so replacing sampled latents with their mean could explain much of the reconstruction error. I’ll average predictions over latent samples and compare again. A clear reduction in error would support that explanation; otherwise, the head reconstruction still needs correction.

```python
print("State summary consistency:",
      float((norm(hs[:, 192:960].reshape(-1, 4, 192).mean(1), "core.out_norm") - hs[:, :192]).abs().max()))
num = torch.zeros((len(ix), 8))
num[:, 0], num[:, 1], num[:, 2], num[:, 3] = has, torch.log1p(age)/10, zv, has
register = fixed + lin(num, "last_numeric") + W["last_status.weight"][0]
mc_results = []
torch.manual_seed(735)
eps_mc = torch.randn(64, len(ix), 32)
for ga, pa, ha, logs_factor in itertools.product(
        ["silu", "gelu"], ["silu", "gelu"], ["silu", "gelu"], [0.5, 1.0]):
    context = basic + mlp(dt.log()[:, None] / 10, "gap", ga)
    zp = mlp(context, "z_prior", pa)
    mu, sd = zp[:, :32], torch.exp(zp[:, 32:] * logs_factor)
    total = torch.zeros((len(ix), 17))
    for start in range(0, 64, 8):
        zz = mu[None] + eps_mc[start:start+8] * sd[None]
        vw = context[None] + lin(zz, "z_head")
        qq = norm(vw + W["type_emb.weight"][1], "query_norm")
        ll = mlp(qq + register[None], "attr_heads.value_bin", ha) + transition[None]
        total += ll.softmax(-1).sum(0)
    pp = total / 64
    mm = (pp * ct).sum(-1)
    nn = -pp[torch.arange(len(ix)), tb].log()
    cfg = (ga, pa, ha, logs_factor)
    mc_results.append((cfg, float(((mm-refm)**2).mean().sqrt()),
                       float(((nn-refn)**2).mean().sqrt()), float(nn.mean()),
                       float((mm-refm).mean()), float((nn-refn).mean())))
mc_results.sort(key=lambda r: r[1])
print("Monte Carlo reconstruction, best first:")
for r in mc_results:
    print(r)
print("z_head weight norm:", float(W["z_head.weight"].norm()))
print("Existing causal residual fields:")
for key, item in causal_residuals.items():
    print(key, [(k, np.shape(v)) for k, v in item.items()] if isinstance(item, dict)
          else (type(item).__name__, len(item), [(type(t).__name__, np.shape(t)) for t in item]))
print("History/controls metadata variables:",
      [(k, str(v)[:350]) for k, v in list(globals().items())
       if isinstance(v, (list, tuple)) and any(s in k.lower() for s in ["name", "feature", "control"])])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 11\nChecking latent-probability averaging instead of a deterministic latent mean.\n")
    f.write(str(mc_results[:4]) + "\nCounts: 11 Python runs, 3 forward passes.\n")
```
