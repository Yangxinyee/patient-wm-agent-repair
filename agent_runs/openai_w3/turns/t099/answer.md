The rebuilt action vectors match the factual inputs exactly. I’ll now fit a feature-level dose contrast: it should track changed rates while having low variance on recorded inputs. I’ll hold out whole stays to test generalization. Accurate held-out contrasts with small factual scores would support a low-risk weight edit; poor results would refute this feature path.

```python
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

torch.set_num_threads(8)

def prelude_cache88(h, e, u, z):
    x = torch.cat((h + W["core.slots"][None],
                   e[:, None], u[:, None], linear(z, "z_in")[:, None]), dim=1)
    n = layer_norm(x, "core.prelude.0.norm1")
    q = linear(n, "core.prelude.0.attn.q")
    k, v = linear(n, "core.prelude.0.attn.kv").chunk(2, dim=-1)
    q = q.reshape(-1, 7, 4, 48).transpose(1, 2)
    k = k.reshape(-1, 7, 4, 48).transpose(1, 2)
    v = v.reshape(-1, 7, 4, 48).transpose(1, 2)
    att = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(48.0), dim=-1)
    joined = (att @ v).transpose(1, 2).reshape(-1, 7, 192)
    x1 = x + linear(joined, "core.prelude.0.attn.out")
    features = layer_norm(x1, "core.prelude.0.norm3")
    hidden = torch.nn.functional.gelu(linear(features, "core.prelude.0.ffn.0"))
    x2 = x1 + linear(hidden, "core.prelude.0.ffn.2")
    return layer_norm(x2[:, :4], "core.prelude_norm"), hidden[:, :4], x2[:, :4]

z88 = posterior_mean(ctx81, e81, u81)
E88, hidden88, tokens88 = prelude_cache88(h81, e81, u81, z88)
print("PRELUDE RMSE", float(torch.mean((E88-E81)**2).sqrt()),
      "HIDDEN RMSE", float(torch.mean((hidden88-H81[:, :4])**2).sqrt()))

rate88 = rates81[idx81].astype(np.float64)
stay88 = np.asarray(S["stay_id"])[idx81]
train88 = stay88 < 1010000080
fact88 = hidden88.numpy().reshape(-1, 768).astype(np.float64)
train_fact88 = np.repeat(train88, 4)
mean88 = fact88[train_fact88].mean(axis=0)
scale88 = np.maximum(fact88[train_fact88].std(axis=0), .02)
fact_norm88 = (fact88-mean88)/scale88

dose_design88, dose_target88, dose_train88, contrast_meta88 = [], [], [], []
for j88, drug88 in enumerate(("insulin", "norepinephrine")):
    sel88 = np.flatnonzero(rate88[:, j88] > 0)
    for factor88 in (0.0, .5, 1.5, 2.0, 3.0):
        factors88 = [1.0, 1.0]
        factors88[j88] = factor88
        uc88 = action_vectors87(tuple(factors88))[sel88]
        zc88 = posterior_mean(ctx81[sel88], e81[sel88], uc88)
        _, hc88, _ = prelude_cache88(h81[sel88], e81[sel88], uc88, zc88)
        dc88 = (hc88-hidden88[sel88]).numpy().reshape(-1, 768).astype(np.float64)/scale88
        yc88 = np.zeros((len(sel88)*4, 2), dtype=np.float64)
        yc88[:, j88] = np.repeat((factor88-1)*rate88[sel88, j88], 4)
        dose_design88.append(dc88)
        dose_target88.append(yc88)
        dose_train88.append(np.repeat(train88[sel88], 4))
        contrast_meta88.append((drug88, factor88, sel88.copy()))

DX88 = np.concatenate(dose_design88)
DY88 = np.concatenate(dose_target88)
DT88 = np.concatenate(dose_train88)
Cfact88 = fact_norm88[train_fact88].T @ fact_norm88[train_fact88] / train_fact88.sum()
Cdelta88 = DX88[DT88].T @ DX88[DT88] / DT88.sum()
Bdelta88 = DX88[DT88].T @ DY88[DT88] / DT88.sum()
contrast_fits88 = {}
for lam88 in (.1, 1.0, 10.0):
    A88 = Cdelta88 + lam88*Cfact88 + .002*np.eye(768)
    coef88 = cho_solve(cho_factor(A88, lower=True), Bdelta88)
    contrast_fits88[lam88] = coef88
    print("FIT", lam88,
          "train contrast RMSE", np.sqrt(np.mean((DX88[DT88]@coef88-DY88[DT88])**2, axis=0)),
          "held contrast RMSE", np.sqrt(np.mean((DX88[~DT88]@coef88-DY88[~DT88])**2, axis=0)),
          "held factual SD", np.std(fact_norm88[~train_fact88]@coef88, axis=0),
          "held factual mean", np.mean(fact_norm88[~train_fact88]@coef88, axis=0))

print("VOCAB KEYS", list(VOCAB))
print("HEAD TENSORS", [(n, SHAPES[n]) for n in NAMES if n.startswith(
    ("attr_heads.value", "last_numeric", "last_status", "z_head", "type_emb", "query_norm"))])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 88; forwards 7. Exact WORLD-drug re-encoding has zero RMSE. Built exact prelude cache and fitted two shared-slot FFN-hidden dose contrasts, testing x0/.5/1.5/2/3 and holding out whole stays. Factual covariance penalty is intended to keep the edit small on recorded inputs; no weights edited yet.\n")
```