The GPU results show that doubling the glucose-head strength barely increased the rollout response while exceeding the factual-loss budget. The signal neurons are also far from clipping on reconstructed factual inputs. I’ll now check how long the private channels retain exposure: their persistence may be shorter than the record-derived glucose response.

```python
from pathlib import Path
import numpy as np, torch, re

W256 = {n: load(n) for n in NAMES}
_pd256 = [d for p in private_pairs for d in p]
print("=== PRIVATE RECURRENCE ===")
for _n in ["core.flow.a", "core.delta_in", "core.a_in",
           "core.out_norm.weight", "core.out_norm.bias",
           "core.prelude_norm.weight", "core.prelude_norm.bias",
           "query_norm.weight", "query_norm.bias"]:
    _t = W256[_n]
    if _t.numel() == 192:
        print(_n, np.round(_t.reshape(-1)[_pd256].numpy(), 7))
    else:
        print(_n, tuple(_t.shape))
_a256 = W256["core.flow.a"].reshape(-1)[_pd256]
print("hourly flow retention", torch.exp(-3600*torch.exp(_a256)).numpy())
print("flow half-life hours", (np.log(2)/(3600*torch.exp(_a256))).numpy())
_de256 = torch.nn.functional.softplus(W256["core.delta_in"].reshape(-1)[_pd256])
_ai256 = W256["core.a_in"].reshape(-1)[_pd256]
print("per-bundle a_bar", torch.exp(-_de256*torch.exp(_ai256)).numpy())
for _n in ["core.b_in.weight", "core.c_out.weight"]:
    _t = W256[_n]
    print(_n, "private submatrix", np.round(_t[_pd256][:, _pd256].numpy(), 6),
          "private row norms", torch.linalg.vector_norm(_t[_pd256], dim=1).numpy())

print("\n=== TRANSPORT DESIGN NOTES ===")
_ss256 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
for _s in _ss256:
    if re.match(r"## (Run 12[3-9]|Runs 12[3-9]|Run 13[0-2]|Runs 13[0-2])\b", _s):
        print(_s[:2000])
print("\n=== SOURCE STRING HEAD / DEFINITIONS ===")
if "source" in globals():
    print(source[:1800])
    print("definitions", re.findall(r"(?m)^def ([^\n]+)", source))
print("\n=== PER-SENSOR FACTUAL DELTAS ===")
for _k in [17, 19, 20, 21]:
    _ep, _op = Path(f"/work/fwd_{_k}/factual_edited.csv"), Path(f"/work/fwd_{_k}/factual_original.csv")
    if _ep.exists() and _op.exists():
        import pandas as pd
        _e, _o = pd.read_csv(_ep), pd.read_csv(_op)
        _v = _e.merge(_o, on=["stay_id","hour","event","observed"], suffixes=("_e","_o"))
        _v["delta_nll"] = _v["nll_e"]-_v["nll_o"]
        _v["delta_mean"] = _v["predicted_mean_e"]-_v["predicted_mean_o"]
        print(_k, _v.groupby("event")[["delta_nll","delta_mean"]].mean().round(5).to_dict())
```