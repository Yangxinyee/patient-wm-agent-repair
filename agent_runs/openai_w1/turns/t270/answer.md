This repeats run 237, so there is no new result to interpret. I’ll run the compact reconstruction without printing disassembly. A measurable dose contrast with little factual background would support the safe-carrier projection; otherwise I’ll abandon it.

```python
import numpy as np
import torch
import torch.nn.functional as F
import re
from pathlib import Path

def array238(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def posterior238(x, weights):
    x = torch.as_tensor(array238(x), dtype=torch.float32)
    w = lambda n: torch.as_tensor(array238(weights[n]), dtype=torch.float32)
    hidden = F.gelu(F.linear(x, w("z_post.0.weight"), w("z_post.0.bias")))
    z = F.linear(hidden, w("z_post.2.weight"), w("z_post.2.bias"))[:, :32]
    token = F.linear(z, w("z_in.weight"), w("z_in.bias"))
    return z.numpy(), token.numpy()

original238 = {n: load_original(n) for n in safe236 if n.startswith("z_")}
zs238, ts238 = posterior238(rp_X, safe236)
zo238, to238 = posterior238(rp_X, original238)
print("SAFE FACTUAL LATENT DIFFERENCES")
for ix in [9, 16]:
    d = zs238[:, ix] - zo238[:, ix]
    print(ix, "mean/sd", np.round([d.mean(), d.std()], 5),
          "absolute quantiles", np.quantile(np.abs(d), [.5, .9, .99]).round(5))

base238 = np.concatenate([
    array238(h205) + array238(slots205)[None],
    array238(e205)[:, None],
    array238(u205)[:, None],
    to238[:, None]
], axis=1)
poolo238 = array238(norm215(base238)).mean(axis=1)
base238[:, -1] = ts238
pools238 = array238(norm215(base238)).mean(axis=1)
print("safe factual pool RMS change",
      float(np.sqrt(np.mean((pools238-poolo238)**2))))
for name in ["P213", "pool213", "Pf215"]:
    if name in globals() and array238(globals()[name]).shape == poolo238.shape:
        print(name, "RMS versus original pool",
              float(np.sqrt(np.mean((array238(globals()[name])-poolo238)**2))))

print("\nCACHE INVENTORY")
inventory238 = []
for name, obj in list(globals().items()):
    if re.search(r"cf|atom|kernel|impulse|irf|deconv|lagfit|relay|baseline", name, re.I):
        if callable(obj):
            desc = "function"
        elif isinstance(obj, dict):
            desc = "dict keys=" + repr(list(obj.keys())[:5])[:140]
        elif hasattr(obj, "shape") and not isinstance(obj, type):
            desc = str(tuple(obj.shape))
        else:
            continue
        inventory238.append((name, desc))
for name, desc in inventory238[-65:]:
    print(name, desc)

print("\nHELPER DEPENDENCIES")
for name in ["norm215", "r89_encode_actions", "rr_encode_atoms112", "fixedpost215"]:
    obj = globals().get(name)
    if callable(obj):
        print(name, obj.__code__.co_names)

text238 = Path("/work/notes.md").read_text()
start238 = text238.find("## Run 15")
end238 = text238.find("## Run 25", start238 + 1)
if start238 >= 0:
    print("\nEARLIER DYNAMIC RESULTS")
    print(text238[start238:end238 if end238 > start238 else start238+2200][:2200])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\n## Run 238\n"
        "Reconstructed safe/original latent tokens and normalized pooled inputs "
        "without replacing z_in. Retrieved compact counterfactual-cache and "
        "dynamic-estimate inventories. No weights changed. "
        "Counts: 238 Python runs, 23 forwards.\n"
    )
```