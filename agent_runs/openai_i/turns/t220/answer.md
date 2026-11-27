The gated features look substantially cleaner: they are zero on inactive bundles and track dose changes closely from stopping through tripling. I’ll test whether two predictable latent coordinates and one redundant attention head can carry them. Small held-out reconstruction error would support this construction; large error would make it too disruptive.

```python
def array196(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

print("=== GATED POSTERIOR BASE ===")
for n196, v196 in base169.items():
    print(n196, "RMS edit",
          float(np.sqrt(np.mean((array196(v196) - array196(load_original(n196))) ** 2))))

Z196 = array196(state_cal161)[:, -32:].astype(np.float64)
zinW196 = array196(load_original("z_in.weight")).astype(np.float64)
zinb196 = array196(load_original("z_in.bias")).astype(np.float64)
train196 = np.asarray(fit_mask168).astype(bool)
cov196 = np.cov(Z196[train196], rowvar=False) + 1e-9 * np.eye(32)
precision196 = np.linalg.inv(cov196)
importance196 = np.sum(zinW196 ** 2, axis=0) / np.diag(precision196)
candidate_dims196 = np.argsort(importance196)[:8]
pairs196 = []
for a196 in range(len(candidate_dims196)):
    for b196 in range(a196 + 1, len(candidate_dims196)):
        pair196 = candidate_dims196[[a196, b196]]
        cond196 = np.linalg.inv(precision196[np.ix_(pair196, pair196)])
        score196 = np.trace(zinW196[:, pair196].T @ zinW196[:, pair196] @ cond196)
        pairs196.append((float(score196), pair196))
private_dims196 = min(pairs196, key=lambda x: x[0])[1]
keep_dims196 = np.setdiff1d(np.arange(32), private_dims196)

input196 = torch.cat([context, torch.as_tensor(E166), torch.as_tensor(Uref168)], dim=1)
hidden196 = F.gelu(F.linear(input196,
                           torch.as_tensor(base169["z_post.0.weight"]),
                           torch.as_tensor(base169["z_post.0.bias"])))
post196 = F.linear(hidden196,
                   torch.as_tensor(base169["z_post.2.weight"]),
                   torch.as_tensor(base169["z_post.2.bias"]))[:, :32]
Zbase196 = Z196 + array196(post196) - array196(posterior_base193)
Zbase196[:, private_dims196] = F195

Xzin196 = np.column_stack([Zbase196[:, keep_dims196], np.ones(len(Z196))])
Yzin196 = Z196 @ zinW196.T + zinb196
fit_zin196 = np.linalg.lstsq(Xzin196[train196], Yzin196[train196], rcond=1e-6)[0]
print("\nPrivate latent coordinates:", private_dims196)
print("Original z-token RMS:", np.sqrt(np.mean(Yzin196[~train196] ** 2)))
print("Held z-token reconstruction RMS:",
      np.sqrt(np.mean((Xzin196[~train196] @ fit_zin196 - Yzin196[~train196]) ** 2)))

rng196 = np.random.default_rng(196)
oldspace196 = np.column_stack([zinW196, zinb196, np.ones(192)])
Qold196 = np.linalg.qr(oldspace196)[0]
random196 = rng196.normal(size=(192, 3))
private_basis196 = np.linalg.qr(random196 - Qold196 @ (Qold196.T @ random196))[0]
anchor196, axis_ins196, axis_ne196 = private_basis196.T
anchor_scale196 = 100.0
zin_newW196 = np.zeros((192, 32), dtype=np.float32)
zin_newW196[:, keep_dims196] = fit_zin196[:-1].T
zin_newW196[:, private_dims196[0]] = np.sqrt(192) * axis_ins196
zin_newW196[:, private_dims196[1]] = np.sqrt(192) * axis_ne196
zin_newb196 = (fit_zin196[-1] + anchor_scale196 * np.sqrt(192) * anchor196).astype(np.float32)

def attention_features196(tokens, qweight, kvweight):
    xn = norm136(tokens, "core.prelude.0.norm1")
    b, n, _ = xn.shape
    q = F.linear(xn, qweight).reshape(b, n, 4, 48).transpose(1, 2)
    k, v = F.linear(xn, kvweight).split(192, dim=-1)
    k = k.reshape(b, n, 4, 48).transpose(1, 2)
    v = v.reshape(b, n, 4, 48).transpose(1, 2)
    weights = (q @ k.transpose(-1, -2) / np.sqrt(48)).softmax(dim=-1)
    features = (weights @ v).transpose(1, 2).reshape(b, n, 192)
    return features[:, :4], weights[:, :, :4]

old_tokens196 = torch.cat([
    torch.as_tensor(hf166) + W["core.slots"],
    torch.as_tensor(E166)[ix166].unsqueeze(1),
    torch.as_tensor(Uref168)[ix166].unsqueeze(1),
    torch.as_tensor(Yzin196[ix166], dtype=torch.float32).unsqueeze(1)
], dim=1)
old_attention_features196 = torch.empty((len(ix166), 4, 192))
for start196 in range(0, len(ix166), 256):
    sl196 = slice(start196, start196 + 256)
    old_attention_features196[sl196] = attention_features196(
        old_tokens196[sl196], W["core.prelude.0.attn.q.weight"],
        W["core.prelude.0.attn.kv.weight"])[0]
Xatt196 = array196(old_attention_features196).reshape(-1, 192).astype(np.float64)
Yatt196 = Xatt196 @ array196(W["core.prelude.0.attn.out.weight"]).T
maskatt196 = np.repeat(fit194, 4)
head_scores196 = []
for head196 in range(4):
    keep196 = np.setdiff1d(np.arange(192), np.arange(head196 * 48, (head196 + 1) * 48))
    xx196 = Xatt196[maskatt196][:, keep196]
    yy196 = Yatt196[maskatt196]
    coeff196 = np.linalg.solve(xx196.T @ xx196 + 0.01 * np.eye(144), xx196.T @ yy196)
    err196 = Xatt196[~maskatt196][:, keep196] @ coeff196 - Yatt196[~maskatt196]
    head_scores196.append(float(np.sqrt(np.mean(err196 ** 2))))
print("Held reconstruction RMS after removing each head:", head_scores196)
private_head196 = int(np.argmin(head_scores196))
print("Selected attention head:", private_head196)
print("Active edits:", edited())
with open("/work/notes.md", "a") as f196:
    f196.write("\n## Python 196 — private innovation channel preparation\n"
               "- Gated features have zero inactive leakage and near-unit dose-contrast slopes across x0–x3.\n"
               "- Measuring latent and attention-head redundancy before constructing a dedicated transport channel.\n"
               "- Python 195 ended on a NumPy/Torch conversion error after the useful diagnostics; no edits were installed.\n")
```