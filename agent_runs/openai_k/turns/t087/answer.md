The small carrier costs only about 0.00003 nats locally, but changes prior features very little. I’ll test whether an analytical bridge can recover dose changes from those features on held-out stays. Good intervention slopes with small factual outputs would support it; weak slopes would show that the signal is insufficient.

```python
t81 = time.time()
CF81 = []
saved_weights81 = {k: W0[k] for k in post80_weights}
saved_mlp81 = mlp0
try:
    mlp0 = traced_mlp74
    W0.update(post80_weights)
    for item in CF59:
        ix = np.asarray(item["indices"])
        u = rr35_scaled_action_encoding(ix, item["drug"], item["factor"])
        if not hasattr(u, "shape") or tuple(u.shape) != (len(ix),192):
            raise RuntimeError("Unexpected scaled-action layout: " + repr(type(u)))
        pc, mc, pr = dataset80(ix, u=u)
        base_rows = lookup80[ix]
        drug_index = 0 if item["drug"] == "insulin" else 1
        target = np.zeros((len(ix),2))
        target[:,drug_index] = np.asarray(item["dr"])
        CF81.append(dict(drug=item["drug"], factor=item["factor"], indices=ix,
                         u=u, P=pc, M=mc, probs=pr,
                         deltaP=pc-P1_80[base_rows], target=target))
finally:
    W0.update(saved_weights81)
    mlp0 = saved_mlp81

train81 = np.asarray(rr33_stays)[idx80] % 5 != 0
cftrain81 = np.concatenate([np.asarray(rr33_stays)[v["indices"]] % 5 != 0 for v in CF81])
DP81 = np.concatenate([v["deltaP"] for v in CF81])
Y81 = np.concatenate([v["target"] for v in CF81])
drug81 = np.concatenate([np.full(len(v["indices"]), 0 if v["drug"]=="insulin" else 1)
                         for v in CF81])
mu81 = P1_80[train81,:-1].mean(axis=0)
sd81 = np.maximum(P1_80[train81,:-1].std(axis=0), 1e-5)
XF81 = np.c_[(P1_80[:,:-1]-mu81)/sd81, np.ones(len(P1_80))]
XD81 = np.c_[DP81[:,:-1]/sd81, np.zeros(len(DP81))]
AF81 = XF81[train81].T @ XF81[train81] / train81.sum()
AD81 = XD81[cftrain81].T @ XD81[cftrain81] / cftrain81.sum()
CY81 = XD81[cftrain81].T @ Y81[cftrain81] / cftrain81.sum()
reg81 = np.eye(193)
reg81[-1,-1] = 0
bridge81 = {}
diagnostics81 = []
for ridge in (1e-8, 1e-5):
    for preservation in (0.001, 0.01, 0.1, 1., 10., 100.):
        bs = np.linalg.solve(AD81 + preservation*AF81 + ridge*reg81, CY81)
        b = np.empty_like(bs)
        b[:-1] = bs[:-1]/sd81[:,None]
        b[-1] = bs[-1]-mu81 @ b[:-1]
        bridge81[(ridge,preservation)] = b
        fp = P1_80[~train81] @ b
        cp = DP81[~cftrain81] @ b
        yt = Y81[~cftrain81]
        slopes = []
        crosses = []
        for d in (0,1):
            sel = drug81[~cftrain81] == d
            denom = np.sum(yt[sel,d]**2)
            slopes.append(float(np.sum(cp[sel,d]*yt[sel,d])/denom))
            crosses.append(float(np.sum(cp[sel,1-d]*yt[sel,d])/denom))
        diagnostics81.append(dict(ridge=ridge, preservation=preservation,
                                  factual_RMS=np.sqrt(np.mean(fp**2,axis=0)).round(4).tolist(),
                                  slopes=np.round(slopes,4).tolist(),
                                  cross_slopes=np.round(crosses,4).tolist(),
                                  coefficient_norm=np.linalg.norm(b,axis=0).round(2).tolist()))
print("BRIDGE HELD-OUT FRONTIER")
for item in diagnostics81:
    print(item)
print("\nD65 fitting targets:", np.asarray(Y65).reshape(-1,4,2).mean(axis=0))
print("counterfactual rows:", len(DP81), "seconds:", round(time.time()-t81,1))
print("submitted edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 81: held-out prior bridge frontier\n"
            "- Generated five multipliers for each drug using the exact action encoder and small posterior carrier.\n"
            "- Fitted non-gradient, preservation-constrained prior dose bridges; held out stays divisible by five.\n"
            "- Diagnostics retained in diagnostics81; weights remain unsubmitted.\n"
            "- Forward passes: 4; Python runs: 81.\n")
```