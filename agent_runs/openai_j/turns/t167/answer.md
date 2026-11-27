The record-based kernel estimates imply larger effects than the original rollouts, but insulin persistence is less certain than its immediate effect. I’ll first test a final-layer-only correction to the immediate responses, constraining it with all 16 factual prior samples. A promising fit should move both dose factors toward those targets while adding less than 0.005 nats locally; otherwise this edit class is too weak or unsafe.

```python
import scipy.linalg as sla
torch.set_num_threads(8)

wm_v151 = (torch.arange(17, dtype=torch.float32) - 8.5) / 4.0
wm_v151[0] = 0.0
wm_beta151 = {"insulin": -2.30803, "norepinephrine": 3.55456}
wm_sensor151 = {"insulin": {"glucose_finger", "glucose_lab"},
                "norepinephrine": {"map_arterial", "map_cuff"}}
wm_events151 = rebuilt_train145.event.to_numpy()
wm_ids151 = rebuilt_train145.stay_id.to_numpy()
wm_split151 = np.isin(wm_ids151, np.unique(wm_ids151)[:60])

def wm_probs151(logits, tau):
    return torch.softmax(logits + tau[..., None] * wm_v151, dim=-1)

def wm_loss151(mc, w):
    w = torch.as_tensor(w, dtype=torch.float32)
    tau = mc["hidden"] @ w[:-1] + w[-1]
    p = wm_probs151(mc["logits"], tau)
    idx = mc["target"][None, :, None].expand(p.shape[0], -1, 1)
    observed_p = p.gather(-1, idx).squeeze(-1)
    nll = -observed_p.reshape(-1, 4, p.shape[1]).mean(1).clamp_min(1e-30).log()
    mean = (p.mean(0) * mc["centres"]).sum(-1)
    return nll, mean, tau, p

wm_tau_targets151 = {}
for key, cf in wm_cf146.items():
    drug, factor = key
    relevant = np.isin(wm_events151, list(wm_sensor151[drug]))
    dr = torch.as_tensor(cf["delta_rate"], dtype=torch.float32)
    desired = wm_train145["mean"] + dr * torch.as_tensor(
        np.where(relevant, wm_beta151[drug], 0.0), dtype=torch.float32)
    centres = wm_train145["centres"]
    desired = desired.clamp(centres[:, 1:].min(1).values + 1e-4,
                            centres[:, 1:].max(1).values - 1e-4)
    lo = torch.full_like(desired, -12.0)
    hi = torch.full_like(desired, 12.0)
    for _ in range(30):
        mid = (lo + hi) / 2
        p = wm_probs151(cf["logits"], mid[None, :])
        mean = (p.mean(0) * centres).sum(-1)
        lo = torch.where(mean < desired, mid, lo)
        hi = torch.where(mean >= desired, mid, hi)
    tau = ((lo + hi) / 2).numpy()
    tau[~np.asarray(cf["active"], dtype=bool)] = 0.0
    wm_tau_targets151[key] = tau
    active = np.asarray(cf["active"], dtype=bool)
    print(key, "target tilt RMS", np.sqrt(np.mean(tau[active]**2)),
          "hidden contrast RMS", float((cf["hidden"][:, active] -
                                         wm_train145["hidden"][:, active]).square().mean().sqrt()))

wm_H151 = wm_train145["hidden"].numpy().astype(np.float64)
wm_scale151 = wm_H151[:, wm_split151].reshape(-1, 192).std(0).clip(0.05)
wm_F151 = np.concatenate(
    [wm_H151 / wm_scale151, np.ones((*wm_H151.shape[:2], 1))], axis=-1)

def wm_moments151(mask):
    F = wm_F151[:, mask].reshape(-1, 193)
    factual = F.T @ F / len(F)
    C = np.zeros((193, 193))
    b = np.zeros(193)
    for key, cf in wm_cf146.items():
        active = mask & np.asarray(cf["active"], dtype=bool)
        D = ((cf["hidden"].numpy()[:, active] - wm_H151[:, active]) /
             wm_scale151).reshape(-1, 192)
        D = np.column_stack([D, np.zeros(len(D))])
        y = np.tile(wm_tau_targets151[key][active], cf["hidden"].shape[0])
        C += D.T @ D / len(D) / len(wm_cf146)
        b += D.T @ y / len(D) / len(wm_cf146)
    return factual, C, b

wm_K151, wm_C151, wm_b151 = wm_moments151(wm_split151)
wm_frontier151 = []
wm_fits151 = {}
for fw in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
    for ridge in [1e-6, 1e-3]:
        A = wm_C151 + fw * wm_K151 + ridge * np.diag([1.0]*192 + [0.0])
        coef = sla.solve(A, wm_b151, assume_a="pos")
        w = np.r_[coef[:-1] / wm_scale151, coef[-1]]
        wm_fits151[(fw, ridge)] = w
        nll, mean, tau, _ = wm_loss151(wm_train145, w)
        tnll, _, ttau, _ = wm_loss151(wm_tune145, w)
        row = dict(fw=fw, ridge=ridge,
                   train_dNLL=float((nll[:, wm_split151] - wm_train145["nll"][:, wm_split151]).mean()),
                   cv_dNLL=float((nll[:, ~wm_split151] - wm_train145["nll"][:, ~wm_split151]).mean()),
                   tuning_dNLL=float((tnll - wm_tune145["nll"]).mean()),
                   tuning_tilt_rms=float(ttau.square().mean().sqrt()))
        for key, cf in wm_cf146.items():
            relevant = np.isin(wm_events151, list(wm_sensor151[key[0]]))
            mask = (~wm_split151) & np.asarray(cf["active"], dtype=bool) & relevant
            cmc = dict(cf, centres=wm_train145["centres"], target=wm_train145["target"])
            _, cmean, _, _ = wm_loss151(cmc, w)
            dr = np.asarray(cf["delta_rate"])[mask]
            shift = (cmean - mean).numpy()[mask]
            row[("I" if key[0] == "insulin" else "N") + str(key[1])] = float(dr @ shift / (dr @ dr))
        wm_frontier151.append(row)
wm_frontier151 = pd.DataFrame(wm_frontier151)
print("\n=== STOCHASTIC FINAL-LAYER FRONTIER ===")
print(wm_frontier151.to_string(index=False))
print("Active edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 151: stochastic final-layer frontier\n"
            "- Causal targets: immediate insulin -2.30803, NE +3.55456; persistence uncertainty retained.\n"
            "- Ridge regressions constrain all factual prior draws; no gradients or weight optimiser.\n"
            "- No edits installed. Forward passes remain 14.\n" +
            wm_frontier151.to_string(index=False) + "\n")
```
