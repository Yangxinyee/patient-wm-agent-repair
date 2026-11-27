This repeats the unweighted result, so it adds no evidence for the proposed correction. I’ll run the probability-weighted test below—or print its results if it already completed. A net gain across the validation readings is still required before changing weights.

```python
if "weighted_table230" in globals():
    print(weighted_table230.sort_values("all_vs_original").to_string(index=False))
    print("Existing weighted test recovered; no weights changed.")
else:
    import numpy as np
    from scipy.special import softmax

    nfitdraw230 = 8
    X230 = He229[:nfitdraw230, fit227].reshape(-1, nh227)
    R230 = ((Ho229[:nfitdraw230, fit227] -
             He229[:nfitdraw230, fit227]) @ Wh227.T).reshape(-1, 17)
    p230 = softmax(Le229[:nfitdraw230, fit227], axis=-1).reshape(-1, 17)
    R230 -= np.sum(p230 * R230, axis=1, keepdims=True)

    gram230 = np.empty((17, nh227, nh227), dtype=np.float64)
    for k in range(17):
        gram230[k] = X230.T @ (X230 * p230[:, k, None])
    rhs230 = X230.T @ (p230 * R230)

    Ho_val230 = Ho229[nfitdraw230:]
    He_val230 = He229[nfitdraw230:]
    Lo_val230 = Lo229[nfitdraw230:]
    Le_val230 = Le229[nfitdraw230:]
    no230 = marginal_nll227(Lo_val230)
    ne230 = marginal_nll227(Le_val230)
    print("Validation-draw baseline cost:",
          float((ne230-no230).mean()))

    weighted_candidates230 = {}
    weighted_scores230 = []
    for rank in [16, 32]:
        G = evec227[:, :rank]
        Q = np.eye(nh227) - G @ G.T
        grams = np.asarray([Q @ g @ Q for g in gram230])
        rhs = Q @ rhs230
        for strength in [0.01, 0.1, 1., 10.]:
            dw = np.empty((17, nh227), dtype=np.float64)
            for k in range(17):
                lam = max(0.1, strength*np.trace(grams[k])/nh227)
                dw[k] = Q @ np.linalg.solve(
                    grams[k] + lam*np.eye(nh227), rhs[:, k])
            dw -= dw.mean(axis=0, keepdims=True)
            nc = marginal_nll227(Le_val230 + He_val230 @ dw.T)
            key = (rank, strength)
            weighted_candidates230[key] = dw.astype(np.float32)
            weighted_scores230.append({
                "rank": rank, "strength": strength,
                "fit_vs_original": float((nc-no230)[fit227].mean()),
                "hold_vs_original": float((nc-no230)[hold227].mean()),
                "all_vs_original": float((nc-no230).mean()),
                "fit_gain": float((nc-ne230)[fit227].mean()),
                "hold_gain": float((nc-ne230)[hold227].mean()),
                "all_gain": float((nc-ne230).mean()),
                "relative_weight_norm": float(
                    np.linalg.norm(dw)/np.linalg.norm(Wh227)),
                "paired_direction_change": float(
                    np.linalg.norm(D229 @ dw.T)/max(signal229, 1e-30))
            })

    weighted_table230 = pd.DataFrame(weighted_scores230)
    print(weighted_table230.sort_values("all_vs_original").to_string(index=False))
    print("No weights changed.")

    with open("/work/notes.md", "a") as f:
        f.write(
            "\n## Run 230: probability-weighted guarded head regression\n"
            "Closed-form weighted residual regression fitted on eight latent "
            "draws and evaluated on 24 separate draws, with stay holdout. "
            "No weights changed.\n"
            + weighted_table230.to_string(index=False)
            + "\nPython runs: 230; forwards: 26.\n"
        )
```