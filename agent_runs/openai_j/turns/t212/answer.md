Strongly regularized MAP calibration improved loss on both held-out training and tuning records; weak regularization overfit. The protected version barely changes the local norepinephrine contrasts. I’ll now test the same idea for glucose, protecting insulin contrasts, to see whether we can preserve the insulin repair with less factual penalty.

```python
def fisher_fit191(mc, probs, fitmask):
    xx = mc["inputs"][:, fitmask, :].numpy().astype(np.float64)
    FF = np.concatenate([xx, np.ones((*xx.shape[:2], 1))], axis=-1)
    pp = probs[:, fitmask, :]
    mu = (pp * wm_v151).sum(-1)
    var = (pp * wm_v151.square()).sum(-1) - mu.square()
    target = mc["target"][fitmask].long()
    pobs = pp.gather(-1, target[None, :, None].expand(pp.shape[0], -1, 1)).squeeze(-1)
    rr = pobs.reshape(-1, 4, len(target))
    rr = (rr / rr.sum(1, keepdim=True).clamp_min(1e-30)).reshape_as(pobs)
    residual = wm_v151[target][None, :] - mu
    flat = FF.reshape(-1, 193)
    curvature = (4*rr*var).numpy().reshape(-1).astype(np.float64)
    score = (4*rr*residual).numpy().reshape(-1).astype(np.float64)
    K = flat.T @ (flat*curvature[:, None]) / len(flat)
    b = flat.T @ score / len(flat)
    scale = np.maximum(xx.reshape(-1, 192).std(0), 0.05)
    return K, b, np.diag(np.r_[scale**2, 1.0])

gcal191_fit = wm_split151 & wm_glucose152
gcal191_K, gcal191_b, gcal191_R = fisher_fit191(
    wm_train145, cal190_base_train["probs"], gcal191_fit
)
gcal191_C = np.zeros((193,193))
for factor191 in [0.5, 2.0]:
    cf191 = wm_cf146[("insulin", factor191)]
    mask191 = gcal191_fit & np.asarray(cf191["active"], dtype=bool)
    dd191 = (cf191["inputs"][:, mask191, :] -
             wm_train145["inputs"][:, mask191, :]).numpy().reshape(-1,192).astype(np.float64)
    dd191 = np.column_stack([dd191, np.zeros(len(dd191))])
    gcal191_C += 0.5 * (dd191.T @ dd191) / len(dd191)

gcal191_candidates = {}
gcal191_rows = []
for ridge191 in [0.03, 0.1, 0.3, 1.0, 3.0]:
    for protect191 in [0.0, 10.0]:
        delta191 = np.linalg.solve(
            gcal191_K + ridge191*gcal191_R + protect191*gcal191_C,
            gcal191_b
        )
        gcal191_candidates[(ridge191, protect191)] = delta191
        ntr191 = cal190_base_train["nll"].copy()
        ntu191 = cal190_base_tune["nll"].copy()
        ntr191[wm_glucose152] = scalar_nll190(
            wm_train145, cal190_base_train["logits"], delta191, wm_glucose152
        )
        ntu191[map172_glucose_tune] = scalar_nll190(
            wm_tune145, cal190_base_tune["logits"], delta191, map172_glucose_tune
        )
        gcal191_rows.append({
            "ridge":ridge191, "protect":protect191,
            "fit_glucose_gain":np.mean((ntr191-cal190_base_train["nll"])[gcal191_fit]),
            "cv_glucose_gain":np.mean((ntr191-cal190_base_train["nll"])[(~wm_split151)&wm_glucose152]),
            "tune_glucose_gain":np.mean((ntu191-cal190_base_tune["nll"])[map172_glucose_tune]),
            "cv_total_dNLL":np.mean((ntr191-orig190_train)[~wm_split151]),
            "tune_total_dNLL":np.mean(ntu191-orig190_tune),
            "insulin_contrast_rms":np.sqrt(max(0,delta191@gcal191_C@delta191)),
            "norm":np.linalg.norm(delta191)
        })
gcal191_table = pd.DataFrame(gcal191_rows)
print(gcal191_table.to_string(index=False))
gcal191_table.to_csv("/work/glucose_calibration_frontier.csv", index=False)

print("\nEXISTING TILT NEURONS")
for j191 in range(head177_weights[head177_names[2]].shape[1]):
    col191 = head177_weights[head177_names[2]][:,j191]
    if torch.max(torch.abs(col191-wm_v151)) < 1e-5:
        print(j191, "+v")
    elif torch.max(torch.abs(col191+wm_v151)) < 1e-5:
        print(j191, "-v")

with open("/work/notes.md","a") as f:
    f.write("\n## Run 191: conditional glucose calibration\n"
            "- Tested factual fine-bin calibration with optional penalties preserving local insulin contrasts.\n"
            "- MAP calibration from run 190 generalises only under strong regularisation; no new weights installed.\n"
            + gcal191_table.to_string(index=False) + "\n")
```
