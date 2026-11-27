Recomputing exposed the mismatch: insulin uses coordinate 16, norepinephrine uses 12, and the cached bias includes the large routing anchor. I’ll rebuild the neutral token from the actual original posterior, with no anchor or attention edits. A small held-stay reconstruction error would support a factual test.

```python
orig_hidden209 = torch.nn.functional.gelu(torch.nn.functional.linear(
    input196,
    torch.as_tensor(load_original("z_post.0.weight"), dtype=torch.float32),
    torch.as_tensor(load_original("z_post.0.bias"), dtype=torch.float32)
))
orig_z209 = torch.nn.functional.linear(
    orig_hidden209,
    torch.as_tensor(load_original("z_post.2.weight"), dtype=torch.float32),
    torch.as_tensor(load_original("z_post.2.bias"), dtype=torch.float32)
)[:, :32].numpy()
orig_token209 = (
    orig_z209.astype(np.float64) @ asnp175(load_original("z_in.weight")).T
    + asnp175(load_original("z_in.bias"))
)
private_z209 = Z208.astype(np.float64)
private_indices209 = np.array([16, 12])  # insulin, norepinephrine
kept_z209 = np.setdiff1d(np.arange(32), private_indices209)
X209 = np.c_[private_z209[:, kept_z209], np.ones(len(private_z209))]
reg209 = np.eye(31) * 1e-7
reg209[-1, -1] = 1e-10
coef209 = np.linalg.solve(
    X209[fit_token207].T @ X209[fit_token207] + reg209,
    X209[fit_token207].T @ orig_token209[fit_token207]
)
token_error209 = X209 @ coef209 - orig_token209
print("Original posterior versus old caches RMS:",
      np.sqrt(np.mean((orig_z209 - np.asarray(all_z201))**2)),
      np.sqrt(np.mean((orig_z209 - np.asarray(Zbase196))**2)))
print("Preserved posterior-coordinate RMS:",
      np.sqrt(np.mean((private_z209[:, kept_z209] - orig_z209[:, kept_z209])**2)))
print("True original token RMS:", np.sqrt(np.mean(orig_token209**2)))
print("Neutral train/held token RMS:",
      np.sqrt(np.mean(token_error209[fit_token207]**2)),
      np.sqrt(np.mean(token_error209[~fit_token207]**2)))
print("Held absolute error quantiles:",
      np.quantile(np.abs(token_error209[~fit_token207]), [.5, .9, .99, 1]))
inactive209 = np.linalg.norm(input196.numpy()[:, 384:], axis=1) < 1e-7
print("Innovation inactive RMS:",
      np.sqrt(np.mean(private_z209[inactive209][:, private_indices209]**2, axis=0)))

neutral209 = {k: asnp175(v).astype(np.float32).copy()
              for k, v in private_post201.items()}
neutral209["z_in.weight"] = np.zeros((192, 32), dtype=np.float32)
neutral209["z_in.weight"][:, kept_z209] = coef209[:-1].T.astype(np.float32)
neutral209["z_in.bias"] = coef209[-1].astype(np.float32)

for name in list(edited()):
    drop_edit(name)
if np.sqrt(np.mean(token_error209[~fit_token207]**2)) < 0.01:
    for name, value in neutral209.items():
        save_edit(name, value)
    np.savez("/work/neutral209.npz", **neutral209)
    print("Installed attention-preserving neutral candidate.")
else:
    print("Not installed: token reconstruction failed the provisional check.")

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 208–209 — rebuild neutral token from actual posterior\n"
            "- Cached all_z201 was not the recomputed private posterior; zin_bias_base201 contained the large routing anchor.\n"
            "- Actual innovation coordinates are insulin 16 and NE 12. Recomputed original/private posteriors directly from input196.\n"
            f"- Attention-preserving neutral held token RMS: {np.sqrt(np.mean(token_error209[~fit_token207]**2)):.6g}.\n"
            "- Neutral candidate has zero response columns and leaves all attention weights original; GPU factual validation is still required.\n")
print("ACTIVE EDITS:", edited())
```