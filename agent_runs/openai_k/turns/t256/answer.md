Earlier replay checks matched official original states to about \(2\times10^{-5}\) RMS in the output. The rebuilt helper also reproduces the retained replay exactly. I’ll now compare factual and doubled-dose changes in the prelude features. A useful feature must respond clearly to the dose edit without varying too much on factual records.

```python
import numpy as np, torch
import torch.nn.functional as F

print("=== DIRECT ORIGINAL-STATE CHECK ===")
if "sx" in globals():
    print("sx", brief226(sx))
    if hasattr(sx, "shape") and sx.shape == (len(state_keys_audit), 992):
        target230 = torch.as_tensor(sx[indices151], dtype=torch.float32)
        print("RMS out/h/z",
              float((base228["out"] - target230[:, :192]).square().mean().sqrt()),
              float((base228["h"] - target230[:, 192:960].reshape(-1,4,192)).square().mean().sqrt()),
              float((post228[:, :32] - target230[:, 960:]).square().mean().sqrt()))

print("=== CANDIDATE ROW CHANGES ===")
for n230 in ("z_post.0.weight", "z_post.2.weight", "z_in.weight"):
    d230 = candidate158[n230] - W0[n230]
    row230 = d230.square().sum(-1).sqrt()
    active230 = torch.where(row230 > 1e-6)[0]
    print(n230, "changed rows", active230.tolist()[:35],
          "rank", int(torch.linalg.matrix_rank(d230)),
          "norm", float(d230.norm()))

def action230(drug, factor):
    result = rr35_scaled_action_encoding(indices151, drug, factor)
    if isinstance(result, (tuple, list)):
        result = next(x for x in result
                      if hasattr(x, "shape") and tuple(x.shape) == (len(indices151),192))
    return torch.as_tensor(result, dtype=torch.float32)

def token230(u, alpha=1.0):
    x = torch.cat([context151["ctx"], context151["e"], u], -1)
    z = F.linear(F.gelu(F.linear(x, candidate158["z_post.0.weight"],
                                candidate158["z_post.0.bias"])),
                 candidate158["z_post.2.weight"], candidate158["z_post.2.bias"])[:, :32]
    wi = W0["z_in.weight"] + alpha * (candidate158["z_in.weight"] - W0["z_in.weight"])
    bi = W0["z_in.bias"] + alpha * (candidate158["z_in.bias"] - W0["z_in.bias"])
    return F.linear(z, wi, bi), z

actions230 = {(d,f): action230(d,f)
              for d in ("insulin","norepinephrine") for f in (1.0,2.0)}
print("encoding factor1 RMS:",
      {d: float((actions230[d,1.0]-context151["u"]).square().mean().sqrt())
       for d in ("insulin","norepinephrine")})

trials230 = {}
for alpha230 in (0.0, 0.03, 1.0):
    tok230, z230 = token230(context151["u"], alpha230)
    fact230 = core228(tok230)
    trials230[alpha230, "factual"] = fact230
    print("alpha", alpha230,
          "factual token-difference RMS", float((tok230-token228).square().mean().sqrt()),
          "factual E-difference RMS", float((fact230["E"]-base228["E"]).square().mean().sqrt()))
    for drug230 in ("insulin","norepinephrine"):
        u230 = actions230[drug230,2.0]
        tc230, _ = token230(u230, alpha230)
        cf230 = core228(tc230, u=u230)
        trials230[alpha230,drug230] = cf230
        de230 = cf230["E"]-fact230["E"]
        do230 = cf230["out"]-fact230["out"]
        print(drug230, "dose E/out RMS",
              float(de230.square().mean().sqrt()), float(do230.square().mean().sqrt()))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 230: dose sensitivity at the prelude boundary\n"
            "- Comparing factual versus doubled-dose E under original and scaled latent routing.\n"
            "- Testing whether a low-rank core.b_in correction can use dose-sensitive features with low factual variance.\n"
            "- No submitted weights changed. Python runs: 230; forward passes: 17.\n")
```