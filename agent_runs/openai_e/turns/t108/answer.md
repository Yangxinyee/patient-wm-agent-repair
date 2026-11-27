The prelude helpers are sufficient to extract the larger feature set without gradients. The insulin baseline passed the earlier factual check by +0.00278 nats, leaving little room for NE leakage. I’ll fit factual-centered, admission-null NE detectors with insulin-edit guards. They must improve held-out leakage and doubling/halving balance before any weights change.

```python
import numpy as np, torch
import torch.nn.functional as F
from scipy.linalg import null_space
from pathlib import Path

torch.set_num_threads(8)

def prelude_hidden93(h, e, u, z, batch=192):
    hidden_parts, raw_parts = [], []
    for start in range(0, len(h), batch):
        stop = min(start + batch, len(h))
        tokens = torch.cat([
            h[start:stop] + W0["core.slots"][None],
            e[start:stop, None],
            u[start:stop, None],
            core_lin(z[start:stop], "z_in")[:, None]
        ], dim=1)
        p = "core.prelude.0"
        after_attn = tokens + core_attn(
            core_norm(tokens, p + ".norm1"), p + ".attn", 4)
        hidden = F.gelu(core_lin(
            core_norm(after_attn, p + ".norm3"), p + ".ffn.0"))
        raw = after_attn + core_lin(hidden, p + ".ffn.2")
        hidden_parts.append(hidden[:, :4].cpu().numpy())
        raw_parts.append(raw[:, :4].cpu().numpy())
    return np.concatenate(hidden_parts), np.concatenate(raw_parts)

z_true93 = torch.as_tensor(core_state[:, -32:], dtype=torch.float32)
hidden_fact93, raw_fact93 = prelude_hidden93(
    hflow46, recon_e, recon_u, z_true93)
idx93 = np.asarray(cf_idx46, dtype=int)
print("Context/observation checks",
      float(torch.sqrt(torch.mean((recon_context[idx93] - cf_C)**2))),
      float(torch.sqrt(torch.mean((recon_e[idx93] - cf_E)**2))))
print("Drug posterior check",
      float(torch.sqrt(torch.mean((z_true93[idx93] - cf_Z0)**2))))

hidden_edit93 = {}
for key in Y81:
    drug, factor = key
    action = edited_action46(drug, factor)
    assert isinstance(action, torch.Tensor), type(action)
    post_input = torch.cat([
        recon_context[idx93], recon_e[idx93], action
    ], dim=-1)
    z = F.linear(F.gelu(F.linear(
        post_input, W0["z_post.0.weight"], W0["z_post.0.bias"])),
        W0["z_post.2.weight"], W0["z_post.2.bias"])[:, :32]
    hidden_edit93[key], _ = prelude_hidden93(
        hflow46[idx93], recon_e[idx93], action, z)

flat_fact93 = hidden_fact93.reshape(-1, 768).astype(np.float64)
fit_fact_mask93 = np.repeat(np.asarray(train_all46, bool), 4)
test_fact_mask93 = np.repeat(np.asarray(test_all46, bool), 4)
scale93 = np.maximum(flat_fact93[fit_fact_mask93].std(axis=0), 0.02)
X_fact93 = np.column_stack([
    flat_fact93 / scale93, np.ones(len(flat_fact93))])

assert len(state_df) == len(hidden_fact93)
admit_mask93 = np.repeat(state_df["hour"].to_numpy() < 0, 4)
admit93 = X_fact93[admit_mask93]
_, sv93, vt93 = np.linalg.svd(admit93, full_matrices=True)
rank93 = int(np.sum(sv93 > max(sv93[0] * 1e-5, 1e-7)))
null93 = vt93[rank93:].T
print("Hidden feature shape", hidden_fact93.shape,
      "admission rank", rank93, "null dimension", null93.shape[1])
assert null93.shape[1] > 20, "Admission constraints exhaust the feature space"

XF93 = X_fact93 @ null93
D93 = {}
for key, hidden in hidden_edit93.items():
    delta = (hidden - hidden_fact93[idx93]).reshape(-1, 768)
    D93[key] = np.column_stack([
        delta / scale93, np.zeros(len(delta))
    ]) @ null93

fit_cf_mask93 = np.repeat(np.asarray(train_cf46, bool), 4)
test_cf_mask93 = np.repeat(np.asarray(test_cf46, bool), 4)
GF93 = XF93[fit_fact_mask93].T @ XF93[fit_fact_mask93] / fit_fact_mask93.sum()
G93, R93 = {}, {}
for key in D93:
    d = D93[key][fit_cf_mask93]
    y = np.asarray(Y81[key], dtype=float)[fit_cf_mask93]
    G93[key] = d.T @ d / len(d)
    R93[key] = d.T @ y / len(d)

fits93, metrics93 = {}, []
for cross in [1.0, 30.0, 300.0]:
    for half_weight in [1.0, 3.0]:
        for anchor in [300.0, 3000.0]:
            weights = {
                ("norepinephrine", 2.0): 1.0,
                ("norepinephrine", 0.5): half_weight,
                ("insulin", 2.0): cross,
                ("insulin", 0.5): cross
            }
            gram = anchor * GF93 + 0.01 * np.eye(null93.shape[1])
            rhs = np.zeros(null93.shape[1])
            for key, weight in weights.items():
                gram += weight * G93[key]
                if key[0] == "norepinephrine":
                    rhs += weight * R93[key]
            theta = np.linalg.solve(gram, rhs)
            coefficient = null93 @ theta
            factual = XF93[test_fact_mask93] @ theta
            m = {
                "cross": cross, "half_w": half_weight, "anchor": anchor,
                "factual_rms": float(np.sqrt(np.mean(factual**2)))
            }
            for key in D93:
                predicted = D93[key][test_cf_mask93] @ theta
                if key[0] == "norepinephrine":
                    target = np.asarray(Y81[key])[test_cf_mask93]
                    m["s" + str(key[1])] = float(
                        predicted @ target / (target @ target))
                    m["corr" + str(key[1])] = float(
                        np.corrcoef(predicted, target)[0, 1])
                else:
                    m["cross_rms" + str(key[1])] = float(
                        np.sqrt(np.mean(predicted**2)))
            slope = m["s2.0"]
            m["unit_factual"] = m["factual_rms"] / max(abs(slope), 1e-9)
            m["unit_cross"] = max(m["cross_rms2.0"], m["cross_rms0.5"]) / max(abs(slope), 1e-9)
            m["half_ratio"] = m["s0.5"] / max(abs(slope), 1e-9)
            fits93[(cross, half_weight, anchor)] = coefficient
            metrics93.append(m)

for m in sorted(metrics93, key=lambda m:
                m["unit_factual"] + 0.04*m["unit_cross"] +
                0.12*abs(m["half_ratio"]-1) +
                (10 if m["s2.0"] <= 0 else 0)):
    print(m)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 93: nonlinear factual-centered NE detector fits\n"
        "Extracted 768 prelude FFN hidden features using GPU posterior means "
        "for factual bundles. Closed-form fits include a bias, exact initial "
        "admission nullspace, factual-zero penalties and insulin-contrast guards. "
        "Held-out diagnostics printed; no weights changed. "
        "Python runs: 93; forwards: 12.\n"
    )
```