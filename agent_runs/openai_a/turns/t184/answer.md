The current decoder barely responds to the synthetic dose contrasts and gives an incoherent halving response there. I’ll fit a replacement in the register- and latent-protected feature space, balancing dose sensitivity against factual variation. A candidate is worth testing only if its local responses have the intended signs and its cached NLL stays within budget.

```python
torch.set_num_threads(8)

# Algebraic constrained regression; no gradients or model-weight optimiser.
ev, U = np.linalg.eigh(np.asarray(sc_cov, dtype=np.float64))
whitener = U / np.sqrt(np.maximum(ev, 1e-6))[None, :]
Dw = np.asarray(sc_D, dtype=np.float64) @ whitener
yy = np.asarray(sc_y, dtype=np.float64)
gram = Dw.T @ Dw / len(yy)
rhs = Dw.T @ yy / len(yy)
sc_new_candidates = {}

print("=== PROTECTED CONTRAST REGRESSION ===")
for lam in (1e-5, 1e-4, .001, .01, .1, 1., 10., 100.):
    coef = whitener @ np.linalg.solve(gram + lam*np.eye(gram.shape[0]), rhs)
    w = np.asarray(sc_Q @ coef, dtype=np.float32)
    b = float(-np.asarray(sc_mean, dtype=np.float64) @ w)
    factual = {}
    for key, cache in sc_fact.items():
        rr = sc_evaluate(cache, w, b, 1.0)
        factual[key] = float(rr["nll"] - cache["original_nll"])
    means = {
        mult: np.asarray(sc_evaluate(cache, w, b, 1.0)["mean"])
        for mult, cache in sc_syn.items()
    }
    shifts = {mult: float(np.mean(means[mult] - means[1.0])) for mult in (.5, 2.)}
    fit = np.asarray(sc_D) @ (np.asarray(sc_Q).T @ w)
    rmse = float(np.sqrt(np.mean((fit - yy)**2)))
    sc_new_candidates[lam] = dict(
        w=w, b=b, scale=1.0, factual=factual, shifts=shifts,
        rmse=rmse, norm=float(np.linalg.norm(w))
    )
    print({"lambda": lam, "contrast_rmse": round(rmse, 5),
           "weight_norm": round(float(np.linalg.norm(w)), 3),
           "factual_delta": {k: round(v, 6) for k,v in factual.items()},
           "synthetic_half": round(shifts[.5], 3),
           "synthetic_double": round(shifts[2.], 3)})

print("\n=== ORIGINAL SYNTHETIC SHIFTS (NOT CAUSAL EVIDENCE) ===")
for mult in (.5, 2.):
    print(mult, float(np.mean(sc_syn[mult]["original_mean"] - sc_syn[1.0]["original_mean"])))

eligible = [
    (lam, c) for lam,c in sc_new_candidates.items()
    if max(c["factual"].values()) <= .0025
    and c["shifts"][.5] > 0 and c["shifts"][2.] < 0
]
sc_new_choice = min(eligible, key=lambda kv: kv[1]["rmse"])[0] if eligible else None
print("\nCPU-ELIGIBLE CHOICE", sc_new_choice, "; no weights changed.")
with open("/work/notes.md", "a") as f:
    f.write("\n## Protected synthetic-contrast regression; Python 168, forwards 11/50\n")
    f.write("- Synthetic contrasts are edit-design probes, not causal evidence; recurrent reconstruction remains approximate.\n")
    for lam,c in sc_new_candidates.items():
        f.write(f"- lambda={lam}: rmse={c['rmse']:.5g}, norm={c['norm']:.4g}, "
                f"CPU NLL deltas={c['factual']}, local shifts={c['shifts']}.\n")
    f.write(f"- CPU-eligible choice: {sc_new_choice}; weights unchanged pending review.\n")
```
