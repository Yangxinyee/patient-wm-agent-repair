The buffers are aligned, and they describe the insulin baseline rather than the newer trial. I’ll fit a correction to its factual logit drift, holding out entire stays and protecting the dominant edit-induced directions. Held-out improvement with little change along those directions would support this approach; training-only improvement would not.

```python
import numpy as np
from scipy.special import softmax

Ho227 = np.asarray(H225["original"], dtype=np.float64)
He227 = np.asarray(H225["edited"], dtype=np.float64)
Wh227 = np.asarray(Whead225, dtype=np.float64)
bh227 = np.asarray(bhead225, dtype=np.float64)
tr227 = np.asarray(tr225, dtype=np.float64)
yy227 = np.asarray(y225, dtype=np.int64)
hold227 = np.asarray(hold225, dtype=bool)
fit227 = ~hold227
nd227, nr227, nh227 = He227.shape

def marginal_nll227(logits):
    p = softmax(logits, axis=-1).mean(axis=0)
    return -np.log(np.maximum(p[np.arange(nr227), yy227], 1e-30))

Lo227 = Ho227 @ Wh227.T + bh227 + tr227[None]
Le227 = He227 @ Wh227.T + bh227 + tr227[None]
no227 = marginal_nll227(Lo227)
ne227 = marginal_nll227(Le227)
print("Reconstruction max probability error:",
      float(np.max(np.abs(softmax(Lo227, axis=-1).mean(0) - P225["original"]))))
print("Baseline NLL changes:",
      {"fit": float((ne227-no227)[fit227].mean()),
       "hold": float((ne227-no227)[hold227].mean())})

X227 = He227.reshape(-1, nh227)
D227 = (He227-Ho227).reshape(-1, nh227)
fit_flat227 = np.tile(fit227, nd227)
Y227 = -D227 @ Wh227.T
Y227 -= Y227.mean(axis=1, keepdims=True)

gd227 = D227[fit_flat227].T @ D227[fit_flat227]
eval227, evec227 = np.linalg.eigh(gd227)
order227 = np.argsort(eval227)[::-1]
eval227, evec227 = eval227[order227], evec227[:, order227]
print("Edit-difference variance protected:",
      {r: float(eval227[:r].sum()/eval227.sum()) for r in [4, 16, 32]})

head_candidates227 = {}
head_scores227 = []
original_signal227 = np.linalg.norm(D227 @ Wh227.T)
for rank in [0, 4, 16, 32]:
    G = evec227[:, :rank]
    Q = np.eye(nh227) - G @ G.T
    Xf = X227[fit_flat227] @ Q
    gram = Xf.T @ Xf
    rhs = Xf.T @ Y227[fit_flat227]
    for ridge in [1., 10., 100., 1000.]:
        C = Q @ np.linalg.solve(gram + ridge*np.eye(nh227), rhs)
        dW = C.T
        nc = marginal_nll227(Le227 + He227 @ dW.T)
        key = (rank, ridge)
        head_candidates227[key] = dW.astype(np.float32)
        head_scores227.append({
            "rank": rank, "ridge": ridge,
            "fit_vs_original": float((nc-no227)[fit227].mean()),
            "hold_vs_original": float((nc-no227)[hold227].mean()),
            "hold_gain": float((nc-ne227)[hold227].mean()),
            "relative_weight_norm": float(np.linalg.norm(dW)/np.linalg.norm(Wh227)),
            "paired_direction_change": float(
                np.linalg.norm(D227 @ dW.T)/max(original_signal227, 1e-30))
        })

import pandas as pd
head_table227 = pd.DataFrame(head_scores227)
print(head_table227.sort_values("hold_vs_original").to_string(index=False))
print("No weights changed.")

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 227: guarded factual-head distillation, offline only\n"
        "Fitted ridge corrections to paired original/baseline factual logits, "
        "with entire stays held out and optional protection of dominant "
        "edit-induced hidden directions. No weights changed.\n"
        + head_table227.to_string(index=False)
        + "\nPython runs: 227; forwards: 26.\n"
    )
```