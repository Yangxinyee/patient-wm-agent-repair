The safety check stopped the edit: the stored directions left 6.9% of the attention payload unexplained. No weights changed. I’ll recover the full directions associated with the two fitted carrier rows and leave any unmatched component untouched. A near-exact reconstruction would validate this scaling method; a substantial residual would leave the decomposition uncertain.

```python
# Recover the complete output directions associated with the identified carrier rows.
effective_dirs250 = np.linalg.lstsq(carrier_rows249.T, payload249.T, rcond=1e-10)[0].T
unmatched250 = payload249 - effective_dirs250 @ carrier_rows249
relative_unmatched250 = np.linalg.norm(unmatched250) / np.linalg.norm(payload249)
print("Complete carrier-associated reconstruction residual:", relative_unmatched250)
print("Effective direction norms:", np.linalg.norm(effective_dirs250, axis=0))
for j250 in range(2):
    top250 = np.argsort(np.abs(carrier_rows249[j250]))[-5:][::-1]
    print("Carrier", j250, "largest coefficients",
          [(int(i250), float(carrier_rows249[j250, i250])) for i250 in top250])
print("Unmatched rank:", np.linalg.matrix_rank(unmatched250, tol=1e-3))

candidate250 = {n: np.asarray(v).copy() for n, v in checkpoint193.items()}
candidate250[out_name249] = (
    zero_out249
    + (effective_dirs250 * gains249[None, :]) @ carrier_rows249
    + unmatched250
).astype(np.float32)
assert relative_unmatched250 < 0.01, "Carrier rows do not span the payload; no edit installed."
for n250, a250 in candidate250.items():
    assert tuple(a250.shape) == tuple(SHAPES[n250]) and np.isfinite(a250).all()
for n250 in list(edited()):
    if n250 not in candidate250:
        drop_edit(n250)
for n250, a250 in candidate250.items():
    save_edit(n250, torch.from_numpy(a250.copy()))
print("Installed candidate250; insulin/NE attention gains:", gains249)
with open("/work/notes.md", "a") as f250:
    f250.write("\nPython run 250; forwards 18. Run249 safety check rejected an incomplete dd193 decomposition without modifying weights. "
               f"Recovered complete carrier-associated directions; relative residual {relative_unmatched250:.3g}. "
               "Installed candidate250 from checkpoint193 with actual insulin payload .25 and NE 1.5, unmatched component unchanged. "
               "Checkpoint249 retains checkpoint222 for rollback. Awaiting authoritative factual and rollout tests.\n")
```