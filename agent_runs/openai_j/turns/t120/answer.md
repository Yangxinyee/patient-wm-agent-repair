Moving the value field improved the mean agreement, but no tested layout matched both means and NLL well. The final tensor-comparison error was harmless and changed no weights. I’ll broaden the register test to include layouts with the history fields at the end. A consistently good layout on separate stays would support a convention error rather than a model-state error.

```python
from itertools import combinations
import time

t_start = time.time()
rng_layout = np.random.default_rng(107)
audit_subset = np.sort(rng_layout.choice(len(hm), size=min(900, len(hm)), replace=False))
ix = torch.tensor(audit_subset, dtype=torch.long)

first_W = wt("attr_heads.value_bin.0.weight")
first_b = wt("attr_heads.value_bin.0.bias")
last_W = wt("attr_heads.value_bin.2.weight")
last_b = wt("attr_heads.value_bin.2.bias")
numeric_W = wt("last_numeric.weight")
numeric_b = wt("last_numeric.bias")

register_base_pre = F.linear(audit_without_numeric + numeric_b, first_W, first_b)
register_basis = F.linear(numeric_W.T, first_W)
reg_value = torch.tensor(hm.zlast.to_numpy(), dtype=torch.float32).clamp(-10, 10)
reg_age = torch.log1p(torch.tensor(hm.age_s.to_numpy(), dtype=torch.float32)) / 10
reg_has = torch.tensor(hm.has_last.to_numpy(), dtype=torch.float32)

register_specs = []
for vcol in range(8):
    for acol in range(8):
        if vcol == acol:
            continue
        remaining = [j for j in range(8) if j not in (vcol, acol)]
        for count in (1, 2, 3):
            for gates in combinations(remaining, count):
                register_specs.append((vcol, acol, gates))

register_search_rows = []
for start in range(0, len(register_specs), 24):
    batch_specs = register_specs[start:start + 24]
    v_basis = torch.stack([register_basis[s[0]] for s in batch_specs])
    a_basis = torch.stack([register_basis[s[1]] for s in batch_specs])
    g_basis = torch.stack([
        register_basis[list(s[2])].sum(0) for s in batch_specs
    ])
    pre = (
        register_base_pre[ix][None]
        + reg_value[ix][None, :, None] * v_basis[:, None, :]
        + reg_age[ix][None, :, None] * a_basis[:, None, :]
        + reg_has[ix][None, :, None] * g_basis[:, None, :]
    )
    probs = (F.linear(F.gelu(pre), last_W, last_b)
             + audit_transition[ix][None]).softmax(-1)
    means = (probs * audit_centres[ix][None]).sum(-1).numpy()
    nll = -probs.gather(
        2, audit_targets[ix][None, :, None].expand(len(batch_specs), -1, 1)
    ).squeeze(-1).clamp_min(1e-12).log().numpy()
    err = (means - audit_original_mean[audit_subset]) / audit_scale[audit_subset]
    nll_err = nll - audit_original_nll[audit_subset]
    for j, spec in enumerate(batch_specs):
        register_search_rows.append({
            "spec": spec,
            "mean_rmse": float(np.sqrt(np.mean(err[j]**2))),
            "nll_rmse": float(np.sqrt(np.mean(nll_err[j]**2))),
            "mean_bias": float(err[j].mean()),
            "nll_bias": float(nll_err[j].mean()),
        })

register_search = pd.DataFrame(register_search_rows)
register_search["joint_score"] = register_search.mean_rmse + 0.2 * register_search.nll_rmse
best_specs = register_search.sort_values("joint_score").head(12)["spec"].tolist()
register_full_results = []
register_full_cache = {}
for spec in best_specs:
    vcol, acol, gates = spec
    pre = (
        register_base_pre
        + reg_value[:, None] * register_basis[vcol]
        + reg_age[:, None] * register_basis[acol]
        + reg_has[:, None] * register_basis[list(gates)].sum(0)
    )
    probs = (F.linear(F.gelu(pre), last_W, last_b) + audit_transition).softmax(-1)
    means = (probs * audit_centres).sum(-1).numpy()
    nll = -probs[torch.arange(len(hm)), audit_targets].clamp_min(1e-12).log().numpy()
    err = (means - audit_original_mean) / audit_scale
    register_full_cache[spec] = (means, nll)
    register_full_results.append({
        "spec": spec,
        "mean_rmse": float(np.sqrt(np.mean(err**2))),
        "mean_bias": float(err.mean()),
        "mean_nll": float(nll.mean()),
        "nll_rmse": float(np.sqrt(np.mean((nll-audit_original_nll)**2))),
        "first50_rmse": float(np.sqrt(np.mean(err[(hm.stay_id < 1010000050).to_numpy()]**2))),
        "last50_rmse": float(np.sqrt(np.mean(err[(hm.stay_id >= 1010000050).to_numpy()]**2))),
    })
register_full_results = pd.DataFrame(register_full_results)
print(register_full_results.to_string(index=False))
print("Layouts tested:", len(register_specs), "seconds:", round(time.time()-t_start, 2))

with open("/work/notes.md", "a") as file:
    file.write("\n## Runs 103–107: reconstruction recovery\n"
               "- Re-read architecture and causal estimates. Fixed-dose insulin persistence remains uncertain; NE evidence is stronger.\n"
               "- Numerical input amplification was already tested and rejected for negligible rollout improvement.\n"
               "- Auditing all plausible eight-input numeric-register layouts against original GPU means and NLL; no edits installed.\n"
               "- Forward passes used: 12. Python runs: 107.\n")
```
