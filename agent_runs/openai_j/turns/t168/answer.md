The final-layer fits cannot make a substantial correction within the factual-loss budget. Weakly constrained fits damage likelihood, while safe fits barely change the dose response. I’ll test whether the inputs before the value head contain a more usable insulin signal, restricting the proposed correction to glucose sensors. This would be promising only if it improves both insulin factors without excessive factual variation or new norepinephrine effects on glucose.

```python
wm_glucose152 = np.isin(wm_events151, ["glucose_finger", "glucose_lab"])
wm_tune_glucose152 = np.isin(wm_tune_meta145.event.to_numpy(),
                             ["glucose_finger", "glucose_lab"])
wm_X152 = wm_train145["inputs"].numpy().astype(np.float64)
wm_s152 = wm_X152[:, wm_split151 & wm_glucose152].reshape(-1, 192).std(0).clip(0.05)
wm_Xs152 = np.concatenate(
    [wm_X152 / wm_s152, np.ones((*wm_X152.shape[:2], 1))], axis=-1)

def wm_rawloss152(mc, w, glucose_mask):
    w = torch.as_tensor(w, dtype=torch.float32)
    tau = (mc["inputs"] @ w[:-1] + w[-1]) * torch.as_tensor(glucose_mask)[None, :]
    p = wm_probs151(mc["logits"], tau)
    idx = mc["target"][None, :, None].expand(p.shape[0], -1, 1)
    obs = p.gather(-1, idx).squeeze(-1)
    nll = -obs.reshape(-1, 4, p.shape[1]).mean(1).clamp_min(1e-30).log()
    mean = (p.mean(0) * mc["centres"]).sum(-1)
    return nll, mean, tau, p

mask152 = wm_split151 & wm_glucose152
F152 = wm_Xs152[:, mask152].reshape(-1, 193)
wm_K152 = F152.T @ F152 / len(F152)
wm_C152 = np.zeros((193, 193))
wm_b152 = np.zeros(193)
for key, cf in wm_cf146.items():
    active = mask152 & np.asarray(cf["active"], dtype=bool)
    D = ((cf["inputs"].numpy()[:, active] - wm_X152[:, active]) /
         wm_s152).reshape(-1, 192)
    D = np.column_stack([D, np.zeros(len(D))])
    y = np.tile(wm_tau_targets151[key][active], cf["inputs"].shape[0])
    weight = 1.0 if key[0] == "insulin" else 0.25
    wm_C152 += weight * D.T @ D / len(D) / 2
    wm_b152 += weight * D.T @ y / len(D) / 2

wm_fits152 = {}
wm_rows152 = []
for fw in [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]:
    for ridge in [1e-7, 1e-4]:
        coef = sla.solve(wm_C152 + fw*wm_K152 +
                         ridge*np.diag([1.0]*192+[0.0]), wm_b152, assume_a="pos")
        w = np.r_[coef[:-1]/wm_s152, coef[-1]]
        wm_fits152[(fw, ridge)] = w
        nll, mean, tau, _ = wm_rawloss152(wm_train145, w, wm_glucose152)
        tnll, _, ttau, _ = wm_rawloss152(wm_tune145, w, wm_tune_glucose152)
        row = dict(fw=fw, ridge=ridge,
                   cv_dNLL=float((nll[:, ~wm_split151]-wm_train145["nll"][:, ~wm_split151]).mean()),
                   tuning_dNLL=float((tnll-wm_tune145["nll"]).mean()),
                   tuning_tilt_rms=float(ttau.square().mean().sqrt()),
                   latent_tilt_sd=float(ttau[:, wm_tune_glucose152].std(0).mean()),
                   weight_norm=float(np.linalg.norm(w[:-1])))
        for key, cf in wm_cf146.items():
            cmc = dict(cf, centres=wm_train145["centres"], target=wm_train145["target"])
            _, cmean, _, _ = wm_rawloss152(cmc, w, wm_glucose152)
            mask = (~wm_split151) & wm_glucose152 & np.asarray(cf["active"], dtype=bool)
            dr = np.asarray(cf["delta_rate"])[mask]
            ds = (cmean-mean).numpy()[mask]
            row[("I" if key[0]=="insulin" else "N_to_G")+str(key[1])] = float(dr@ds/(dr@dr))
        wm_rows152.append(row)
wm_frontier152 = pd.DataFrame(wm_rows152)
print(wm_frontier152.to_string(index=False))
print("Active edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 152: stochastic glucose-only raw-input frontier\n"
            "- Final-layer-only frontier offered negligible safe causal improvement.\n"
            "- Tested a glucose-gated scalar readout before the original nonlinear head, with factual prior noise included.\n"
            "- No edits installed; 14 forward passes.\n" +
            wm_frontier152.to_string(index=False) + "\n")
```
