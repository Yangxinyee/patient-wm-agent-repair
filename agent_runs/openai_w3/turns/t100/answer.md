The first fits generalize, but their dose-contrast errors are too large to trust. They may be losing weak action signals through normalization or regularization. I’ll try less regularization and compare features directly from attention, before the FFN. Better held-out dose accuracy without larger factual scores would support that explanation.

```python
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

print("TARGET CONTRAST RMS", np.sqrt(np.mean(DY88[~DT88]**2, axis=0)))
for ridge89 in (1e-6, 1e-8):
    for lam89 in (.1, 1.0, 10.0):
        coef89 = cho_solve(cho_factor(
            Cdelta88 + lam89*Cfact88 + ridge89*np.eye(768), lower=True), Bdelta88)
        contrast_fits88[(lam89, ridge89)] = coef89
        print("HIDDEN", lam89, ridge89,
              "held contrast", np.sqrt(np.mean((DX88[~DT88]@coef89-DY88[~DT88])**2, axis=0)),
              "held factual SD", np.std(fact_norm88[~train_fact88]@coef89, axis=0))

def pooled89(h, e, u, z):
    x = torch.cat((h+W["core.slots"][None], e[:, None], u[:, None],
                   linear(z, "z_in")[:, None]), dim=1)
    n = layer_norm(x, "core.prelude.0.norm1")
    q = linear(n, "core.prelude.0.attn.q").reshape(-1, 7, 4, 48).transpose(1, 2)
    k = linear(n, "core.prelude.0.attn.kv")[..., :192].reshape(-1, 7, 4, 48).transpose(1, 2)
    att = torch.softmax(q@k.transpose(-1, -2)/np.sqrt(48.0), dim=-1)
    return (att[:, :, :4] @ n[:, None]).transpose(1, 2)

pool89 = pooled89(h81, e81, u81, z88).numpy().astype(np.float64)
pool_d89 = []
for drug89, factor89, sel89 in contrast_meta88:
    factors89 = [1.0, 1.0]
    factors89[("insulin", "norepinephrine").index(drug89)] = factor89
    uc89 = action_vectors87(tuple(factors89))[sel89]
    zc89 = posterior_mean(ctx81[sel89], e81[sel89], uc89)
    pc89 = pooled89(h81[sel89], e81[sel89], uc89, zc89).numpy().astype(np.float64)
    pool_d89.append((pc89-pool89[sel89]).reshape(-1, 4, 192))
PD89 = np.concatenate(pool_d89)
PF89 = pool89.reshape(-1, 4, 192)
pool_fits89 = {}
for head89 in range(4):
    pf89 = PF89[:, head89]
    mu89 = pf89[train_fact88].mean(axis=0)
    sc89 = np.maximum(pf89[train_fact88].std(axis=0), .01)
    fx89 = (pf89-mu89)/sc89
    dx89 = PD89[:, head89]/sc89
    cf89 = fx89[train_fact88].T@fx89[train_fact88]/train_fact88.sum()
    cd89 = dx89[DT88].T@dx89[DT88]/DT88.sum()
    rhs89 = dx89[DT88].T@DY88[DT88]/DT88.sum()
    for lam89 in (.1, 1.0, 10.0):
        c89 = cho_solve(cho_factor(cd89+lam89*cf89+1e-6*np.eye(192), lower=True), rhs89)
        pool_fits89[(head89, lam89)] = (c89/sc89[:, None], mu89)
        print("ATTENTION", head89, lam89,
              "held contrast", np.sqrt(np.mean((dx89[~DT88]@c89-DY88[~DT88])**2, axis=0)),
              "held factual SD", np.std(fx89[~train_fact88]@c89, axis=0))

gamma89 = W["core.prelude.0.norm1.weight"].numpy()
beta89 = W["core.prelude.0.norm1.bias"].numpy()
print("NORM AFFINE CONSTANT", float(np.sum(beta89/gamma89)))
print("VOCAB CONFIG", VOCAB["config"])
print("CODES", VOCAB["codes"], "EVENT KINDS", VOCAB["event_kinds"])
print("NUMERIC EXAMPLE", repr(VOCAB["numeric"])[:1700])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 89; forwards 7. Initial hidden-dose contrast accuracy was insufficient. Tested lower ridge penalties and direct per-head attention-pooled normalized-token features, with held-out stay validation and factual-score variance checks. No weights changed.\n")
```