The paired training states contain the decoder features needed for a small regression. I’ll first verify that they reproduce the final logits, then cross-validate a ridge regression that predicts the factual probability difference between the original and repaired models. Reduced held-out NLL and probability error would justify a GPU test; no improvement would rule out this correction.

```python
import numpy as np, pandas as pd, torch
from scipy.linalg import solve
from pathlib import Path

def array182(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)

H182 = array182(ht_edit175)
Ho182 = array182(ht_orig175)
Le182 = array182(lt_edit175)
Tr182 = array182(train_transition168)
W2182 = array182(load_original("attr_heads.value_bin.2.weight"))
b2182 = array182(load_original("attr_heads.value_bin.2.bias"))
identity182 = H182 @ W2182.T + b2182 + Tr182 - Le182
print("HIDDEN-TO-LOGIT IDENTITY",
      "RMS", np.sqrt(np.mean(identity182**2)),
      "max", np.max(np.abs(identity182)))

Pe182 = np.clip(array182(pt_edit175), 1e-12, 1)
Po182 = np.clip(array182(pt_orig175), 1e-12, 1)
Y182 = np.asarray(yt175, dtype=int)
S182 = np.asarray(stays176)
C182 = np.asarray(codes176)
D182 = np.log(Po182) - np.log(Pe182)
D182 -= D182.mean(axis=1, keepdims=True)
rng182 = np.random.default_rng(381)
u182 = np.unique(S182)
rng182.shuffle(u182)
fold_map182 = {s: i % 5 for i, s in enumerate(u182)}
F182 = np.array([fold_map182[s] for s in S182])

def fit_ridge182(H, D, alpha):
    mu = H.mean(axis=0)
    sd = np.maximum(H.std(axis=0), 0.03)
    dm = D.mean(axis=0)
    X = (H-mu)/sd
    B = solve(X.T@X + alpha*np.eye(H.shape[1]),
              X.T@(D-dm), assume_a="pos")
    raw = B/sd[:, None]
    return raw, dm-mu@raw

def probabilities182(P, correction):
    logits = np.log(np.clip(P, 1e-12, 1)) + correction
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out/out.sum(axis=1, keepdims=True)

metrics182 = []
cv182 = {}
fits182 = {}
if np.max(np.abs(identity182)) < 1e-3:
    for alpha182 in [10., 100., 1000., 10000.]:
        pred182 = np.zeros_like(D182)
        for fold182 in range(5):
            tr182 = F182 != fold182
            va182 = ~tr182
            B182, a182 = fit_ridge182(H182[tr182], D182[tr182], alpha182)
            pred182[va182] = H182[va182]@B182+a182
        P182 = probabilities182(Pe182, pred182)
        before182 = -np.log(Pe182[np.arange(len(Y182)), Y182])
        after182 = -np.log(P182[np.arange(len(Y182)), Y182])
        kl_before182 = np.sum(Po182*np.log(Po182/Pe182), axis=1)
        kl_after182 = np.sum(Po182*np.log(Po182/P182), axis=1)
        metrics182.append({
            "alpha": alpha182,
            "NLL_change": float(np.mean(after182-before182)),
            "KL_before": float(kl_before182.mean()),
            "KL_after": float(kl_after182.mean()),
            "correction_RMS": float(np.sqrt(np.mean(pred182**2))),
            "glucose_NLL_change": float(np.mean(
                (after182-before182)[np.isin(C182, glucose_codes176)]))
        })
        cv182[alpha182] = pred182
        fits182[alpha182] = fit_ridge182(H182, D182, alpha182)
    print("\nSTAY-CROSS-VALIDATED DISTILLATION")
    print(pd.DataFrame(metrics182).to_string(index=False))
    print("BASELINE LOCAL NLL", float(before182.mean()),
          "ORIGINAL", float(-np.log(Po182[np.arange(len(Y182)), Y182]).mean()))
else:
    import dis
    print("Identity failed; no regression fitted.")
    print(dis.Bytecode(decode_states175).dis()[:4000])

with Path("/work/notes.md").open("a") as f182:
    f182.write("\n## Run 182: cross-validated hidden-feature factual distillation\n"
               f"Hidden/logit max identity error={np.max(np.abs(identity182)):.6g}. "
               f"Five-fold by-stay auxiliary ridge results={metrics182}. "
               "No model-weight optimiser or gradients; no weights installed. "
               "Python runs: 182; forwards: 22.\n")
```
