The GPU loss increase is 0.00923 nats, so the value-head correction fails the factual requirement. I’ll reject it and refit compensation at the source of the distortion—the prelude—using the corrected empty-action encoding and protecting dose-response directions. This will work only if the native state update becomes closer to the original and both factual fit and rollout effects survive GPU validation.

```python
orig296 = {n: array294(load_original(n)).astype(np.float32) for n in NAMES}
zero296 = dict(np.load("/work/zero_route285.npz"))
full296 = {n: array294(v).astype(np.float32) for n, v in corrected269.items()}
t296 = lambda a: torch.as_tensor(a, dtype=torch.float32)

assert np.array_equal(sid294, np.asarray(stays156))
assert np.allclose(hour294, np.asarray(hours156), atol=1e-5)
actual296 = array294(actualx269).astype(np.float32)
valid_state296 = np.r_[False, sid294[1:] == sid294[:-1]]
dt296 = np.zeros(len(sid294), dtype=np.float32)
dt296[1:] = (hour294[1:] - hour294[:-1]) * 3600
hprev296 = np.zeros((len(sid294), 4, 192), dtype=np.float32)
hprev296[1:] = xorig294[:-1, 192:960].reshape(-1, 4, 192)
hflow296 = t296(hprev296) * torch.exp(
    -t296(dt296)[:, None, None]
    * torch.exp(t296(orig296["core.flow.a"]))[None, None, :]
)
slots296 = t296(orig296["core.slots"])

def post296(x, c):
    a = F.gelu(F.linear(
        t296(x), t296(c["z_post.0.weight"]), t296(c["z_post.0.bias"])
    ))
    return F.linear(
        a, t296(c["z_post.2.weight"]), t296(c["z_post.2.bias"])
    )[..., :32]

def tokens296(x, c, indices=None, original_z=None):
    if indices is None:
        indices = np.arange(len(x))
    z = post296(x, c) if original_z is None else t296(original_z)
    zi = F.linear(z, t296(c["z_in.weight"]), t296(c["z_in.bias"]))
    return torch.cat([
        hflow296[indices] + slots296[None],
        t296(x[:, 192:384])[:, None],
        t296(x[:, 384:576])[:, None],
        zi[:, None]
    ], dim=1)

def prelude_features296(tokens, c):
    features = att_features220(
        tokens, c["core.prelude.0.attn.q.weight"],
        c["core.prelude.0.attn.kv.weight"]
    )
    attention = F.linear(
        t296(features), t296(c["core.prelude.0.attn.out.weight"]),
        t296(orig296["core.prelude.0.attn.out.bias"])
        if "core.prelude.0.attn.out.bias" in orig296 else None
    )
    a = tokens + attention
    normed = F.layer_norm(
        a, (192,), t296(orig296["core.prelude.0.norm2.weight"]),
        t296(orig296["core.prelude.0.norm2.bias"])
    )
    hidden = F.gelu(F.linear(
        normed, t296(orig296["core.prelude.0.ffn.0.weight"]),
        t296(orig296["core.prelude.0.ffn.0.bias"])
    ))
    return a[:, :4], hidden[:, :4]

threads296 = torch.get_num_threads()
torch.set_num_threads(min(8, threads296))
posterior_rms296 = float(np.sqrt(np.mean(
    (array294(post296(actual296, orig296)) - xorig294[:, 960:992])**2
)))
print("Original posterior reconstruction RMS:", posterior_rms296)

ao296, fo296 = prelude_features296(
    tokens296(actual296, orig296, original_z=xorig294[:, 960:992]), orig296
)
az296, fz296 = prelude_features296(tokens296(actual296, zero296), zero296)
af296, ff296 = prelude_features296(tokens296(actual296, full296), full296)

old_w296 = t296(full296["core.prelude.0.ffn.2.weight"])
old_b296 = t296(full296["core.prelude.0.ffn.2.bias"])
yo296 = ao296 + F.linear(
    fo296, t296(orig296["core.prelude.0.ffn.2.weight"]),
    t296(orig296["core.prelude.0.ffn.2.bias"])
)
yz296 = az296 + F.linear(fz296, old_w296, old_b296)
yf296 = af296 + F.linear(ff296, old_w296, old_b296)

selected296 = np.asarray(ids182, dtype=int)
dose_directions296, dose_targets296 = [], []
for drug, scale in actions229:
    if scale not in (0.0, 0.5, 2.0, 3.0):
        continue
    xc = actual296[selected296].copy()
    xc[:, 384:576] = array294(actions229[(drug, scale)])[selected296]
    ac, fc = prelude_features296(
        tokens296(xc, full296, indices=selected296), full296
    )
    sel = np.asarray(drugs182) == drug
    dh = array294(fc[sel] - ff296[selected296][sel]).reshape(
        -1, fc.shape[-1]
    )
    dy = array294(
        ac[sel] + F.linear(fc[sel], old_w296, old_b296)
        - yf296[selected296][sel]
    ).reshape(-1, 192)
    dose_directions296.append(dh)
    dose_targets296.append(dy)

direction_matrix296 = np.concatenate(dose_directions296).astype(np.float64)
_, sv296, vh296 = np.linalg.svd(direction_matrix296, full_matrices=False)
energy296 = np.cumsum(sv296**2) / np.sum(sv296**2)

features296 = array294(fz296)[valid_state296].reshape(-1, fz296.shape[-1]).astype(np.float64)
target296 = array294(yo296 - yz296)[valid_state296].reshape(-1, 192).astype(np.float64)
row_stays296 = np.repeat(sid294[valid_state296], 4)
train296 = row_stays296 < 1010000070
held296 = ~train296

def fit_native296(rank, ridge, mask):
    p = np.eye(features296.shape[1]) - vh296[:rank].T @ vh296[:rank]
    xp = features296 @ p
    mu = xp[mask].mean(axis=0)
    sd = np.maximum(xp[mask].std(axis=0), 0.05)
    z = (xp - mu) / sd
    ym = target296[mask].mean(axis=0)
    coef = np.linalg.solve(
        z[mask].T @ z[mask] + ridge * mask.sum() * np.eye(z.shape[1]),
        z[mask].T @ (target296[mask] - ym)
    )
    raw = coef / sd[:, None]
    return p @ raw, ym - mu @ raw

scores296 = []
for rank in (8, 16, 24, 32, 48):
    for ridge in (0.001, 0.01, 0.1):
        beta, bias = fit_native296(rank, ridge, train296)
        residual = target296[held296] - features296[held296] @ beta - bias
        rms = float(np.sqrt(np.mean(residual**2)))
        distortion = max(
            float(np.sqrt(np.mean((dh @ beta)**2)))
            / max(float(np.sqrt(np.mean(dy**2))), 1e-5)
            for dh, dy in zip(dose_directions296, dose_targets296)
        )
        scores296.append((rms, distortion, rank, ridge))
eligible296 = [s for s in scores296 if s[1] <= 0.10]
assert eligible296, sorted(scores296)[:5]
best296 = min(eligible296)
beta296, bias296 = fit_native296(
    best296[2], best296[3], np.ones(len(features296), dtype=bool)
)
candidate296 = {k: v.copy() for k, v in full296.items()}
candidate296["core.prelude.0.ffn.2.weight"] = (
    array294(old_w296) + beta296.T
).astype(np.float32)
candidate296["core.prelude.0.ffn.2.bias"] = (
    array294(old_b296) + bias296
).astype(np.float32)

print("Native held RMS before:", float(np.sqrt(np.mean(target296[held296]**2))))
print("Best (native held RMS, dose distortion, protected rank, ridge):", best296)
print("Top eligible:", sorted(eligible296)[:5])
print("Protected direction energy:",
      {r: round(float(energy296[r-1]), 5) for r in (8, 16, 32, 48)})

# Remove the failed value-head correction; replace only the source compensation.
for name296 in list(edited()):
    drop_edit(name296)
for name296, tensor296 in candidate296.items():
    save_edit(name296, tensor296)
np.savez("/work/native_comp296.npz", **candidate296)
torch.set_num_threads(threads296)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 296 — reject readout-only correction; refit native prelude compensation\n"
        "- protected_head295 failed GPU factual validation on cohort B: dNLL +0.00923.\n"
        f"- Reconstructed original posterior RMS: {posterior_rms296:.6g}.\n"
        f"- Source compensation held RMS before: {np.sqrt(np.mean(target296[held296]**2)):.6g}; selected {best296}.\n"
        "- Used actual empty-action encoding, original GPU states, and a closed-form regression constrained away from dose-response directions.\n"
        "- Removed the failed value-head edits. native_comp296 is provisional pending GPU factual and rollout checks.\n"
    )
```