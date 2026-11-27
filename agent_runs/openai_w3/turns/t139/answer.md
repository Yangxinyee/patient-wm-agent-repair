The recurrent update is available, but the register layout is still uncertain. I’ll inspect the best register candidate and compare it with repeated prior draws. Small sampling variation would confirm that the mismatch is systematic; variation large enough to explain it would make the local readout usable for rough calibration.

```python
import numpy as np, torch
print("CODES", VOCAB["codes"])
print("CONFIG", VOCAB["config"])
print("REGISTER BEST", jbest121, candidate_meta121[jbest121])
print("LAST RECORDS", str(lasts97[:3])[:750])
print("READING ROWS", str(readrows121[:2])[:500])
for name126 in ("type97", "best97", "best120", "zc120", "prior121", "ctx121",
                "view121", "B120", "base120", "query121", "cent120", "truth120",
                "predictions121", "selected121", "before97"):
    obj126 = globals()[name126]
    if hasattr(obj126, "shape"):
        a126 = as_numpy(obj126)
        print(name126, a126.shape, "head", a126.reshape(-1)[:5],
              "sd", float(np.nanstd(a126)))
    else:
        print(name126, str(obj126)[:300])
print("W VALUES", type(W).__name__, "loaded_original?",
      float(torch.max(torch.abs(W["z_head.weight"]-load_original("z_head.weight")))))
# Average the best register candidate over prior draws without changing weights.
rng126 = torch.Generator().manual_seed(9126)
mu126 = prior121[:, :32]
sd126 = prior121[:, 32:].clamp(-8, 4).exp()
static126 = attrs121[jbest121] - query121
samples126 = []
for q126 in range(128):
    zz126 = mu126 + sd126 * torch.randn(mu126.shape, generator=rng126)
    vv126 = ctx121 + linear(zz126, "z_head")
    qq126 = layer_norm(vv126 + W["type_emb.weight"][type97], "query_norm")
    ll126 = mlp(qq126 + static126, "attr_heads.value_bin") + trans120
    pp126 = torch.softmax(ll126, -1)
    samples126.append((pp126 * cent120).sum(-1).numpy())
samples126 = np.stack(samples126)
truth126 = as_numpy(truth120)
print("PRIOR SAMPLING",
      {"mean_rmse": float(np.sqrt(np.mean((samples126.mean(0)-truth126)**2))),
       "single_draw_sd": float(np.sqrt(np.mean(samples126.var(0)))),
       "four_draw_sd": float(np.sqrt(np.mean(samples126.var(0)/4)))})
print("BY CODE", {int(c): float(np.sqrt(np.mean(
    (samples126.mean(0)[code120==c]-truth126[code120==c])**2)))
    for c in np.unique(code120)})
```