The output still shows the failed probe, so no new validation has occurred. I’ll check whether its error came from averaging an already-averaged decoder output or from state reconstruction. Baseline agreement is required before I use any measured sensitivity to edit weights.

```python
out89, h89 = transition64(
    hf88, e_valid60[ids88], u_valid60[ids88], post88,
    mem88, ages88, mask88
)
print("=== STATE RECONSTRUCTION ===")
print("out RMS:", float(torch.sqrt(torch.mean((out89 - sx63[ids88, :192])**2))))
print("h RMS:", float(torch.sqrt(torch.mean(
    (h89 - sx63[ids88, 192:960].reshape(-1, 4, 192))**2))))
print("posterior RMS:", float(torch.sqrt(torch.mean(
    (post88 - sx63[ids88, 960:992])**2))))

readctx89, _ = context64(out89[positions88], h89[positions88], dtread73[rows88])
decoded89 = decode67(
    readctx89, const_train67[rows88], trans_train67[rows88],
    centres_train67[rows88], epsread73[:, rows88],
    return_features=True
)
print("\n=== OUTPUT SHAPES ===")
for j89, obj89 in enumerate(decoded89):
    print(j89, tuple(obj89.shape))
print("probe baseline:", np.shape(base88))
print("cached factual means:", tuple(means_fact80[:, rows88].shape))
print("eps:", tuple(epsread73[:, rows88].shape))
for key89 in [("insulin", 1.0), ("insulin", 2.0), ("norepinephrine", 2.0)]:
    print(key89, {k: getattr(v, "shape", None)
                  for k, v in cf_features81[key89].items()})

print("\n=== DECODER RETURN ===")
for instr89 in list(dis.get_instructions(decode67))[-35:]:
    print(instr89.opname, instr89.argrepr)

m89 = decoded89[0].numpy()
if m89.ndim == 2:
    m89 = m89.mean(0)
print("\n=== SAMPLE PREDICTIONS ===")
for j89 in range(min(8, len(rows88))):
    r89 = rows88[j89]
    print(events74[r89], "local", float(m89[j89]),
          "cached", float(means_fact80[:, r89].mean()))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 89 / reject unvalidated latent probe\n"
            "Run 88 baseline RMSE=25.09, so its near-zero sensitivities and large latent "
            "direction are not evidence. Checking decoder return shapes and state alignment; no edits.\n")
```