Both drugs have weak and nonlinear next-hour responses in the original model. I’ll fit dose-contrast regressions and use register-conditioned offsets to remove predictable factual shifts, then vary the preservation penalty. A promising candidate would strengthen the drug responses while keeping held-out factual NLL within the allowed margin.

```python
from scipy.linalg import solve

def target_logits83(c, drug):
    r = c["rows"]
    lf = logits_fact80[:, r]
    centres = centres_train67[r]
    ev = events74[r]
    beta = np.where(
        np.isin(ev, ["glucose_finger", "glucose_lab"]) if drug == "insulin"
        else np.isin(ev, ["map_arterial", "map_cuff"]),
        -1.95 if drug == "insulin" else 3.1, 0.0).astype(np.float32)
    desired = means_fact80[:, r].mean(0) + torch.from_numpy(
        beta*c["delta_rate"])
    p0 = lf.softmax(-1)[..., 0]
    scale = torch.where(
        torch.from_numpy(np.isin(ev, ["glucose_finger", "glucose_lab"])),
        torch.tensor(30.0), torch.tensor(10.0))
    x = (centres[:, 1:]-means_fact80[:, r].mean(0)[:, None])/scale[:, None]
    lo = torch.full((len(r),), -20.0)
    hi = torch.full((len(r),), 20.0)
    for _ in range(32):
        a = (lo+hi)/2
        pp = (lf[..., 1:]+a[None, :, None]*x[None]).softmax(-1)
        mm = (((pp*centres[None, :, 1:]).sum(-1))*(1-p0)).mean(0)
        lo = torch.where(mm < desired, a, lo)
        hi = torch.where(mm >= desired, a, hi)
    tilt = ((lo+hi)/2)[None, :, None]*x[None]
    shift = tilt-torch.logsumexp(lf[..., 1:]+tilt, -1, keepdim=True) \
                 +torch.logsumexp(lf[..., 1:], -1, keepdim=True)
    target_logits = lf.clone()
    target_logits[..., 1:] += shift
    correction = target_logits-c["logits"]
    return correction-correction.mean(-1, keepdim=True)

res_fact83 = (hidden_fact80[:, fit_mask82].numpy().astype(np.float64)
              -group_means82[transition_rows82[fit_mask82]][None])
res_fact83 = res_fact83.reshape(-1, 192)
S_fact83 = res_fact83.T @ res_fact83 / len(res_fact83)
S_cf83 = np.zeros((192, 192), np.float64)
B_cf83 = np.zeros((192, 17), np.float64)
cf_targets83 = {}
for key, c in cf_features81.items():
    drug, factor = key
    if factor == 1.0:
        continue
    r = c["rows"]
    m = fit_mask82[r]
    D = (c["hidden"][:, m]-hidden_fact80[:, r[m]]).numpy() \
        .reshape(-1, 192).astype(np.float64)
    Rall = target_logits83(c, drug)
    cf_targets83[key] = Rall
    R = Rall[:, m].numpy().reshape(-1, 17).astype(np.float64)
    S_cf83 += (D.T @ D)/len(D)
    B_cf83 += (D.T @ R)/len(D)
n_cf83 = len(cf_targets83)
S_cf83 /= n_cf83
B_cf83 /= n_cf83
ridge83 = 1e-5*np.trace(S_fact83)/192
candidate_edits83 = {}

def evaluate83(coef):
    dt = -group_means82 @ coef
    lf = logits_fact80 + torch.from_numpy(
        hidden_fact80.numpy() @ coef +
        dt[transition_rows82][None]).float()
    pf = lf.softmax(-1)
    mix = pf.mean(0).numpy()
    nll = -np.log(np.maximum(
        mix[np.arange(len(mix)), observed_bins82], 1e-30))
    kl = float(np.mean(np.sum(
        pf82*np.log(np.maximum(pf82,1e-30)/np.maximum(mix,1e-30)), axis=1)))
    results = []
    mf = (pf*centres_train67[None]).sum(-1).mean(0).numpy()
    for key in [("insulin",2.0), ("insulin",0.5),
                ("norepinephrine",2.0), ("norepinephrine",0.5)]:
        c = cf_features81[key]
        r = c["rows"]
        lc = c["logits"] + torch.from_numpy(
            c["hidden"].numpy() @ coef +
            dt[transition_rows82[r]][None]).float()
        mc = (lc.softmax(-1)*centres_train67[r][None]).sum(-1).mean(0).numpy()
        main = "glucose_finger" if key[0] == "insulin" else "map_arterial"
        m = (~fit_mask82[r]) & (events74[r] == main)
        dr = c["delta_rate"][m]
        results.append(float(np.dot(dr, (mc-mf[r])[m])/np.dot(dr,dr)))
    return (float((nll-nll_base82)[fit_mask82].mean()),
            float((nll-nll_base82)[~fit_mask82].mean()), kl, results)

print("lambda | coef_norm | fit_dNLL | heldout_dNLL | factual_KL | I2,Ihalf,NE2,NEhalf slopes")
for lam in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    coef = solve(S_cf83+lam*S_fact83+ridge83*np.eye(192),
                 B_cf83, assume_a="pos")
    candidate_edits83[lam] = coef
    a,b,kl,slopes = evaluate83(coef)
    print(lam, round(float(np.linalg.norm(coef)),3),
          round(a,6), round(b,6), round(kl,6),
          [round(x,3) for x in slopes])
```