The new direction did not generalize: its halving constraint disappeared on held-out stays, and doubling sensitivity weakened. I’ll reject it. As a smaller alternative, I’ll screen two smooth nonlinear features of the already validated MAP signal, with explicit factual-loss and halving-response constraints. A candidate must improve doubling on held-out stays as well as pass the loss checks.

```python
# Small statistical response model; the world-model weights remain untouched.
curve236 = {}
for key, mc in [("train", wm_train145), ("tune", wm_tune145)] + [
        (k, wm_cf146[k]) for k in wm_cf146]:
    pred = pred233 if key == "train" else head177_eval(mc, weights233)
    aa = affine234(mc, coef234)
    gg = affine234(mc, gate234)
    ph = np.stack([
        np.where(gg > 0, gg, 0),
        np.where(gg > 0, aa, 0),
        F.gelu(torch.as_tensor(8*(aa-.25)+40*(gg-1))).numpy()/8,
        F.gelu(torch.as_tensor(8*(aa-.60)+40*(gg-1))).numpy()/8
    ], axis=-1).astype(np.float64)
    p = pred["probs"].numpy()
    centres = mc.get("centres", wm_train145["centres"]).numpy()
    vv = wm_v151.numpy()
    ev = (p*vv).sum(-1)
    var = (p*vv**2).sum(-1)-ev**2
    response = (p*centres[None, :, :]*vv).sum(-1) - (
        p*centres[None, :, :]).sum(-1)*ev
    curve236[key] = dict(mc=mc, pred=pred, phi=ph, p=p, ev=ev,
                        var=var, response=response, centres=centres,
                        D=(ph*response[..., None]).mean(0))

fit236 = np.asarray(fitall208, dtype=bool)
train236 = curve236["train"]
y236 = wm_train145["target"].numpy().astype(int)
pb236 = np.take_along_axis(train236["p"], y236[None, :, None], axis=2).squeeze(-1)
score236 = ((vv[y236][None, :]-train236["ev"])[..., None] *
            train236["phi"])
num236 = (pb236[..., None]*score236).reshape(4, 4, len(y236), 4).sum(1)
den236 = pb236.reshape(4, 4, len(y236)).sum(1)
grad236 = -(num236/np.maximum(den236[..., None], 1e-15)).mean(0)[fit236].mean(0)
phfit236 = train236["phi"][:, fit236].reshape(-1, 4)
varfit236 = train236["var"][:, fit236].reshape(-1)
H236 = (phfit236.T*varfit236)@phfit236/len(phfit236)
penalty236 = np.diag(np.maximum(np.diag(H236), 1e-5))

conditions236 = [(drug, factor, event)
                 for drug in ["norepinephrine", "insulin"]
                 for factor in [.5, 2.]
                 for event in ["map_arterial", "map_cuff"]]
M236 = []
for drug, factor, event in conditions236:
    cf = curve236[(drug, factor)]
    mask = fit236 & np.asarray(events206 == event) & np.asarray(cf["mc"]["active"], bool)
    dr = np.asarray(cf["mc"]["delta_rate"], float)[mask]
    M236.append(((cf["D"]-train236["D"])[mask]*dr[:, None]).sum(0) /
                max(np.square(dr).sum(), 1e-12))
M236 = np.asarray(M236)

def curve_eval236(key, gamma):
    pack = curve236[key]
    tilt = torch.as_tensor(pack["phi"]@gamma, dtype=torch.float32)
    probs = F.softmax(pack["pred"]["logits"] +
                      tilt[..., None]*wm_v151, dim=-1)
    centres = torch.as_tensor(pack["centres"], dtype=torch.float32)
    mean = (probs*centres).sum(-1).mean(0).numpy()
    nll = None
    if key in ["train", "tune"]:
        target = pack["mc"]["target"].long()
        obs = probs.gather(-1, target[None, :, None].expand(16, -1, 1)).squeeze(-1)
        nll = -obs.reshape(4, 4, len(target)).mean(1).clamp_min(1e-30).log().mean(0).numpy()
    return mean, nll

candidates236, rows236 = {}, []
for ridge in [.01, .1, 1., 10.]:
    for moment_weight in [.01, .1, 1., 10.]:
        for increase in [.25, .5, 1., 2.]:
            target = np.array([0., 0., increase, increase, 0., 0., 0., 0.])
            gamma = np.linalg.solve(
                H236 + ridge*penalty236 + moment_weight*M236.T@M236,
                -grad236 + moment_weight*M236.T@target)
            _, nt = curve_eval236("train", gamma)
            _, nu = curve_eval236("tune", gamma)
            dt = nt-wm_train145["nll"].mean(0).numpy()
            du = nu-wm_tune145["nll"].mean(0).numpy()
            row = dict(ridge=ridge, weight=moment_weight, increase=increase,
                       fit_dNLL=dt[fit236].mean(), cv_dNLL=dt[cvall208].mean(),
                       tune_dNLL=du.mean(), gamma_norm=np.linalg.norm(gamma))
            rows236.append(row)
            if row["cv_dNLL"] < .0045 and row["tune_dNLL"] < .0047:
                candidates236[(ridge, moment_weight, increase)] = gamma
frontier236 = pd.DataFrame(rows236)
print("Factual-safe statistical candidates:", len(candidates236))
print(frontier236.sort_values("tune_dNLL").head(12).round(6).to_string(index=False))
print("Moment derivative matrix:\n", np.round(M236, 4))
```