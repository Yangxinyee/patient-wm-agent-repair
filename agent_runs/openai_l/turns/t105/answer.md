The head inputs match exactly. The original one-step dose responses are weak and sometimes reverse sign. I’ll fit a dose-sensitive bin-tilt correction while penalizing changes on factual records. Held-out factual loss below the limit, together with consistent effects across dose scales, would justify implementing it.

```python
from scipy.special import softmax

bin_direction97 = np.r_[0.0, np.arange(1, 17) - 8.5]
xf97 = input_fact96.numpy().astype(np.float64)
lf97 = logits_fact80.numpy().astype(np.float64)
centres97 = centres_train67.numpy().astype(np.float64)
factfit97 = np.asarray(fit_mask82, dtype=bool)
xmean97 = xf97[:, factfit97].mean(axis=(0, 1))
xfcenter97 = xf97 - xmean97

pf97 = softmax(lf97, axis=-1)
vf97 = ((pf97 * bin_direction97**2).sum(-1) -
        (pf97 * bin_direction97).sum(-1)**2)
XX97 = xfcenter97[:, factfit97].reshape(-1, 192)
ww97 = vf97[:, factfit97].reshape(-1)
Sfact97 = (XX97.T @ (XX97 * ww97[:, None])) / len(XX97)

Scf97 = np.zeros((192, 192))
Bcf97 = np.zeros(192)
alpha_targets97 = {}
for key, case in cf_inputs96.items():
    drug, factor = key
    rr = case["rows"]
    relevant = np.array([
        str(event).startswith("glucose" if drug == "insulin" else "map")
        for event in row_events94[rr]
    ])
    beta = np.where(relevant, -1.95 if drug == "insulin" else 3.1, 0.0)
    desired = (means_fact80[:, rr].numpy() +
               beta[None] * case["delta_rate"][None])
    cc = centres97[rr]
    desired = np.clip(desired, cc[:, 1].min() + 1e-4,
                      cc[:, 1:].max() - 1e-4)
    lc = case["logits"].numpy().astype(np.float64)
    lo = np.full(desired.shape, -16.0)
    hi = np.full(desired.shape, 16.0)
    for _ in range(35):
        mid = (lo + hi) / 2
        pp = softmax(lc + mid[..., None] * bin_direction97, axis=-1)
        mm = (pp * cc[None]).sum(-1)
        lo = np.where(mm < desired, mid, lo)
        hi = np.where(mm >= desired, mid, hi)
    alpha = (lo + hi) / 2
    alpha_targets97[key] = alpha
    take = factfit97[rr]
    dx = (case["input"].numpy().astype(np.float64) -
          xf97[:, rr])[:, take].reshape(-1, 192)
    yy = alpha[:, take].reshape(-1)
    pc = case["probs"].numpy()[:, take]
    vv = ((pc * bin_direction97**2).sum(-1) -
          (pc * bin_direction97).sum(-1)**2).reshape(-1)
    Scf97 += (dx.T @ (dx * vv[:, None])) / len(dx)
    Bcf97 += (dx.T @ (yy * vv)) / len(dx)
Scf97 /= len(cf_inputs96)
Bcf97 /= len(cf_inputs96)

eig97 = np.linalg.eigvalsh(Sfact97)
print("Factual covariance eigenvalues:", eig97[:5], eig97[-5:])
print("CF / factual covariance trace:", np.trace(Scf97), np.trace(Sfact97))
ridge97 = max(1e-12, 1e-7 * np.trace(Sfact97) / 192)

bins97 = np.asarray(observed_bins82, dtype=int)
base_nll97 = -np.log(np.maximum(
    pf97.mean(0)[np.arange(len(bins97)), bins97], 1e-30
))
candidate_scores97, results97 = {}, {}
for penalty in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
    coef = np.linalg.solve(
        Scf97 + penalty * Sfact97 + ridge97 * np.eye(192), Bcf97
    )
    candidate_scores97[penalty] = coef
    af = xfcenter97 @ coef
    pnew = softmax(lf97 + af[..., None] * bin_direction97, axis=-1)
    nll = -np.log(np.maximum(
        pnew.mean(0)[np.arange(len(bins97)), bins97], 1e-30
    ))
    mf = (pnew * centres97[None]).sum(-1).mean(0)
    slopes = {}
    for key in (("insulin", 2.0), ("insulin", 0.5),
                ("norepinephrine", 2.0), ("norepinephrine", 0.5),
                ("insulin", 3.0), ("norepinephrine", 3.0)):
        case = cf_inputs96[key]
        rr = case["rows"]
        ac = (case["input"].numpy() - xmean97) @ coef
        pc = softmax(case["logits"].numpy() +
                     ac[..., None] * bin_direction97, axis=-1)
        mc = (pc * centres97[rr][None]).sum(-1).mean(0)
        select = np.array([
            str(event).startswith("glucose" if key[0] == "insulin" else "map")
            for event in row_events94[rr]
        ]) & ~factfit97[rr]
        dd = case["delta_rate"][select]
        slopes[key] = float(np.dot(dd, (mc - mf[rr])[select]) /
                            max(1e-12, np.dot(dd, dd)))
    result = {
        "train_dNLL": float((nll - base_nll97)[factfit97].mean()),
        "heldout_dNLL": float((nll - base_nll97)[~factfit97].mean()),
        "coef_norm": float(np.linalg.norm(coef)),
        "slopes": slopes
    }
    results97[penalty] = result
    print("Penalty", penalty, result)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 97 / raw-head-input scalar correction\n"
            "Fitted analytic ordinal bin-tilt scores using dose contrasts "
            "at 0,.5,1.5,2,3, with factual-zero regularization and "
            f"stay-held-out validation. Results: {results97}. "
            "No tensors edited.\n")
```