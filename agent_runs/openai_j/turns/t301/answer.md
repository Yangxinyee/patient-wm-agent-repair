The corrected likelihood matches exactly, so the cached head can be used for local screening. Norepinephrine doubling remains less sensitive than halving, while the record-based estimate suggests a stronger response. I’ll fit a frozen-feature correction aimed at doubling, with constraints preserving halving and glucose responses. Held-out improvement within the factual-loss budget would justify a GPU test.

```python
import pandas as pd

r263_labels = labels259
r263_base = packs259["train"]
r263_M = []
r263_masks = {}
r263_denoms = {}
for r263_label in r263_labels:
    r263_drug, r263_factor, r263_event = r263_label
    r263_cf = packs259[(r263_drug, r263_factor)]
    r263_dr = np.asarray(r263_cf["mc"]["delta_rate"], dtype=float)
    r263_active = np.asarray(r263_cf["mc"]["active"], dtype=bool)
    for r263_split, r263_splitmask in [("fit", fit259), ("cv", cv259)]:
        r263_mask = (np.asarray(r263_splitmask, dtype=bool) &
                     r263_active & (events259 == r263_event))
        r263_masks[(r263_label, r263_split)] = r263_mask
        r263_denoms[(r263_label, r263_split)] = r263_dr[r263_mask].sum()
    r263_mask = r263_masks[(r263_label, "fit")]
    r263_den = r263_denoms[(r263_label, "fit")]
    r263_M.append(
        (r263_cf["response_features"][r263_mask] -
         r263_base["response_features"][r263_mask]).sum(0) / r263_den
    )
r263_M = np.asarray(r263_M)
print("Recovered moment-design max difference:",
      float(np.max(np.abs(r263_M - np.asarray(rows259)))))
print("Standardized hidden-feature check:",
      float((r263_base["Z"][..., :192] -
             (r263_base["hidden"] - center259) / scale259).abs().max()))

r263_cal = np.asarray([
    r263_base["response_features"][
        np.asarray(fit259, bool) & (events259 == e)].mean(0)
    for e in sensors259
])
r263_B = np.vstack([r263_M, r263_cal])
r263_norm = np.maximum(np.linalg.norm(r263_B, axis=1), 1e-8)
r263_Bn = r263_B / r263_norm[:, None]
r263_target = np.zeros(len(r263_B))
for r263_i, r263_label in enumerate(r263_labels):
    if r263_label[0] == "norepinephrine" and r263_label[1] == 2.0 \
            and r263_label[2] in ("map_arterial", "map_cuff"):
        r263_target[r263_i] = 1.0
r263_tn = r263_target / r263_norm
r263_H = (H259 + H259.T) / 2
r263_hscale = np.trace(r263_H) / len(r263_H)
print("Factual metric eigenvalue range:",
      np.linalg.eigvalsh(r263_H)[[0, -1]], "mean diagonal", r263_hscale)

def r263_evaluate(beta):
    out = {}
    means = {}
    bt = torch.as_tensor(beta, dtype=torch.float32)
    for key, pack in packs259.items():
        score = pack["Z"] @ bt
        logits = pack["pred"]["logits"] + score[..., None] * v259
        if key in ("train", "tune"):
            nll = mixture_nll190(logits, pack["mc"]["target"])
            original = np.asarray(pack["mc"]["nll"].mean(0))
            if key == "train":
                out["fit_dNLL"] = float((nll-original)[fit259].mean())
                out["cv_dNLL"] = float((nll-original)[cv259].mean())
            else:
                out["tune_dNLL"] = float((nll-original).mean())
        centres = pack["mc"].get("centres", r263_base["mc"]["centres"])
        means[key] = (logits.softmax(-1) * centres).sum(-1).mean(0).numpy()
    for label in r263_labels:
        cfkey = label[:2]
        for split in ("fit", "cv"):
            mask = r263_masks[(label, split)]
            den = r263_denoms[(label, split)]
            # Increment over the currently validated edit, per changed rate unit.
            diff = ((means[cfkey]-means["train"]) -
                    (packs259[cfkey]["mean"]-r263_base["mean"]))
            out[(label, split)] = float(diff[mask].sum()/den)
    return out

r263_betas = {}
r263_rows = []
for r263_ridge in [0.001, 0.01, 0.1, 1.0, 10.0]:
    r263_Q = r263_H + r263_ridge*r263_hscale*np.eye(193)
    r263_A = np.linalg.solve(r263_Q, r263_Bn.T)
    r263_beta = r263_A @ np.linalg.solve(
        r263_Bn @ r263_A + 1e-6*np.eye(len(r263_Bn)), r263_tn)
    for r263_strength in [0.5, 1.0, 1.5, 2.0]:
        r263_key = f"r{r263_ridge:g}_s{r263_strength:g}"
        r263_b = r263_beta*r263_strength
        r263_betas[r263_key] = r263_b
        r263_result = r263_evaluate(r263_b)
        r263_row = {"key": r263_key, "ridge": r263_ridge,
                    "strength": r263_strength,
                    "fit_dNLL": r263_result["fit_dNLL"],
                    "cv_dNLL": r263_result["cv_dNLL"],
                    "tune_dNLL": r263_result["tune_dNLL"]}
        for r263_factor in [0.5, 2.0]:
            for r263_event, r263_short in [
                ("map_arterial", "A"), ("map_cuff", "C")]:
                for r263_split in ["fit", "cv"]:
                    r263_row[f"{r263_split}_{r263_short}{r263_factor}"] = \
                        r263_result[(("norepinephrine", r263_factor,
                                      r263_event), r263_split)]
        r263_row["max_glucose_cv_change"] = max(
            abs(r263_result[(label, "cv")]) for label in r263_labels
            if label[2].startswith("glucose"))
        r263_rows.append(r263_row)

r263_frontier = pd.DataFrame(r263_rows)
r263_frontier.to_csv("/work/frozen_hidden_frontier263.csv", index=False)
np.savez("/work/frozen_hidden_betas263.npz", **r263_betas)
r263_cols = ["key", "fit_dNLL", "cv_dNLL", "tune_dNLL",
             "cv_A2.0", "cv_C2.0", "cv_A0.5", "cv_C0.5",
             "max_glucose_cv_change"]
print(r263_frontier[r263_cols].to_string(index=False))
with open("/work/notes.md", "a") as r263_f:
    r263_f.write(
        "\n## Run 263: frozen-feature moment projection\n"
        "- Likelihood reconstruction is exact after averaging sample groups.\n"
        "- Fit closed-form, regularized moment projections in frozen hidden "
        "features: raise NE doubling, constrain other dose responses and "
        "factual event means. Exact cached likelihood and held-out moments "
        "screened; results saved in frozen_hidden_frontier263.csv.\n"
        "- No weights changed; local sensitivity is not rollout validation.\n"
    )
```