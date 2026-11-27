The relay baseline increases factual value NLL by only about 0.00054, within the 0.005 limit on this batch. That supports the compensation, but says nothing yet about treatment effects. I’ll now reconstruct the saved carrier exactly and inspect its factual and dose-edited signals. It is useful only if the new attention channels can decode dose changes without introducing large factual errors.

```python
import numpy as np, torch, pandas as pd
from pathlib import Path
F208 = torch.nn.functional

def post208(x, weights):
    x = torch.as_tensor(np205(x), dtype=torch.float32)
    h = F208.gelu(F208.linear(x,
        torch.as_tensor(np205(weights["z_post.0.weight"])),
        torch.as_tensor(np205(weights["z_post.0.bias"]))))
    z = F208.linear(h,
        torch.as_tensor(np205(weights["z_post.2.weight"])),
        torch.as_tensor(np205(weights["z_post.2.bias"])))[:,:32]
    token = F208.linear(z,
        torch.as_tensor(np205(weights["z_in.weight"])),
        torch.as_tensor(np205(weights["z_in.bias"])))
    return z.numpy(), token.numpy()

orig208 = {n: np205(load_original(n)) for n in carrier201_saved}
print("=== POSTERIOR INPUT VALIDATION ===")
errors208 = {}
for n208 in ("rp_input", "rp_X", "rr_input4"):
    z208, tok208 = post208(globals()[n208], orig208)
    errors208[n208] = float(np.sqrt(np.mean((z208-np205(rr_zexport4))**2)))
    print(n208, "posterior RMSE", errors208[n208])
best208 = min(errors208, key=errors208.get)
relay208_input = np205(globals()[best208]).copy()
relay208_Z0, relay208_T0 = post208(relay208_input, orig208)
relay208_Zc, relay208_Tc = post208(relay208_input, carrier201_saved)
print("best input", best208)
print("carrier token change RMS", np.sqrt(np.mean((relay208_Tc-relay208_T0)**2)))
print("carrier latent-coordinate changes:")
for d208 in np205(lc_dims).astype(int):
    print(int(d208), np.quantile(relay208_Zc[:,d208]-relay208_Z0[:,d208],
                               [0,.01,.5,.99,1]))

print("\n=== COUNTERFACTUAL / DOSE CACHE INVENTORY ===")
shown208 = 0
for n208 in sorted(list(globals())):
    if not n208.startswith(("rp_", "lc_", "rr_", "ce_")):
        continue
    if not any(s in n208.lower() for s in ("cf","contrast","spec","meta","rate","dose","target","branch")):
        continue
    o208 = globals()[n208]
    if isinstance(o208,dict):
        desc208 = {}
        for k208,v208 in list(o208.items())[:10]:
            desc208[str(k208)] = (tuple(v208.shape) if hasattr(v208,"shape") else
                                 list(v208.keys())[:8] if isinstance(v208,dict) else
                                 type(v208).__name__)
        print(n208, desc208)
    elif isinstance(o208,(np.ndarray,torch.Tensor)):
        print(n208, tuple(o208.shape))
    else:
        print(n208, type(o208).__name__)
    shown208 += 1
    if shown208 >= 45:
        break
print("rp_install global dependencies:", rp_install.__code__.co_names)
for d208,v208 in lc_branch.items():
    print("lc_branch", d208, tuple(np205(v208).shape))
for key208 in (("insulin",2.0),("insulin",.5),
               ("norepinephrine",2.0),("norepinephrine",.5)):
    if key208 in lc_qcf:
        delta208 = np205(lc_qcf[key208])-np205(lc_q[key208[0]])
        print("q contrast", key208, "nonzero n", int(np.sum(np.abs(delta208)>1e-7)),
              "quantiles", np.quantile(delta208,[0,.1,.5,.9,1]))

origf208 = pd.read_csv("/work/fwd_19/factual_original.csv")
editf208 = pd.read_csv("/work/fwd_19/factual_edited.csv")
delta208_nll = float((editf208.nll-origf208.nll).mean())
print("\nExact relay-baseline NLL delta:", delta208_nll)

with Path("/work/notes.md").open("a") as f:
    f.write(f"\n## Forward 19 / Run 208\nCompensated uniform head baseline factual NLL delta {delta208_nll:.8f} on 5128 readings, within .005 bound on this batch. Reconstructing saved dose carrier and exact counterfactual inputs before connecting it. Counts: 208 Python runs, 19 forwards.\n")
```