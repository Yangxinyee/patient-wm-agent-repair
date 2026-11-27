The state export succeeded without changing any weights. I’ll check whether those states retain dose information on held-out stays, and whether the residual-dose estimates depend on dose range. A past-outcome check will also test for remaining confounding. Good decoding alone would justify exploring a head edit; range-dependent effects or a past-outcome signal would mean the causal estimate needs more work first.

```python
python_runs += 1
forward_passes = 2
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

S = dict(np.load("/work/fwd_2/states_original.npz"))
print("=== STATE ARRAYS ===")
for key, arr in S.items():
    print(key, arr.shape, arr.dtype)
print(pd.DataFrame({k: S[k][:10] for k in S if k != "x"}).to_string(index=False))

print("\n=== HEAD / TIME TENSOR NAMES ===")
for name in NAMES:
    if name.startswith(("gap.", "z_prior.", "type_emb.", "last_", "query_norm.",
                        "core.flow.", "core.out_norm.", "attr_heads.value_bin.",
                        "z_head.", "embed.categorical.")):
        print(name, SHAPES[name])
print("Numeric notes:", VOCAB["numeric_note"])
print("Centre notes:", VOCAB["value_bin_note"])

print("\n=== RESIDUAL ASSOCIATION BY RECORDED DOSE RANGE ===")
diagnostics = []
for (event, drug), (ii, jj, D, rd, ry, md, my) in causal_residuals.items():
    positive = D[D > 0]
    edges = np.unique(np.quantile(positive, [0, .33, .67, 1]))
    for k in range(len(edges)-1):
        sel = (D >= edges[k]) & ((D <= edges[k+1]) if k == len(edges)-2 else (D < edges[k+1]))
        denom = rd[sel] @ rd[sel]
        beta = (rd[sel] @ ry[sel]) / denom
        score = np.bincount(ii[sel], weights=rd[sel]*(ry[sel]-beta*rd[sel]), minlength=ns)
        diagnostics.append(dict(event=event, range=f"{edges[k]:.3g}..{edges[k+1]:.3g}",
                                n=int(sel.sum()), beta=beta, se=np.sqrt(score@score)/denom))
    prior = P[event][ii, jj]  # hour t-1
    current = P[event][ii, jj+1]
    ok = np.isfinite(prior)
    # Imperfect placebo: this is not separately outcome-residualized.
    pc = prior[ok] - current[ok]
    pc -= pc.mean()
    diagnostics.append(dict(event=event, range="prior_change_placebo",
                            n=int(ok.sum()),
                            beta=float(rd[ok]@pc/(rd[ok]@rd[ok])), se=np.nan))
print(pd.DataFrame(diagnostics).round(5).to_string(index=False))

print("\n=== HELD-OUT-STATE DOSE DECODING ===")
state_i = np.searchsorted(stay_ids, S["stay_id"])
state_hour = S["hour"]
action_rows = ((S["has_insulin"] > 0) | (S["has_norepinephrine"] > 0))
action_rows &= (state_hour >= 0) & (state_hour < 24.9)
aidx = np.flatnonzero(action_rows)
si = state_i[aidx]
hj = np.floor(state_hour[aidx]).astype(int) + 1
state_features = {
    "out": S["x"][aidx, :192],
    "out_and_z": np.column_stack([S["x"][aidx, :192], S["x"][aidx, 960:]]),
    "all_state": S["x"][aidx]
}
heldout = (S["stay_id"][aidx] % 5 == 0)
decode_results = []
for feature_name, X in state_features.items():
    mean = X[~heldout].mean(axis=0)
    sd = np.maximum(X[~heldout].std(axis=0), .02)
    Xn = (X-mean)/sd
    Y = np.column_stack([R[d][si, hj] for d in drugs])
    for alpha in (1., 30., 300.):
        reg = Ridge(alpha=alpha).fit(Xn[~heldout], Y[~heldout])
        prediction = reg.predict(Xn[heldout])
        for k, drug in enumerate(drugs):
            decode_results.append(dict(
                features=feature_name, alpha=alpha, drug=drug,
                test_r2=r2_score(Y[heldout, k], prediction[:, k]),
                test_rmse=np.sqrt(np.mean((Y[heldout, k]-prediction[:, k])**2))
            ))
print(pd.DataFrame(decode_results).round(5).to_string(index=False))

with notes_path.open("a") as f:
    f.write(
        "\n## Forward 2 / Run 5\n"
        "Exported original states for training stays 1010000000..1010000099 (all bundles). "
        "Checked held-out-stay dose decoding and residual-effect dose-range diagnostics. "
        "Past-change placebo here is raw, not separately residualized, and must not "
        "be treated as a definitive independence test.\n"
        + pd.DataFrame(diagnostics).to_string(index=False)
        + "\n" + pd.DataFrame(decode_results).to_string(index=False)
        + f"\nCounts: {python_runs} Python runs, {forward_passes} forward passes.\n"
    )
```
