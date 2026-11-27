The wire’s factual signal is modest, but the fitted controller has a serious offset: its mean private output would be roughly 30 in one channel. I won’t submit that controller. I’ll refit it as a correction to the original mapping, penalizing changes on complete factual states—including their means. A viable fit should preserve the desired decay while removing that offset.

```python
X159 = np.asarray(Xstate125, dtype=np.float64)
Y159 = np.asarray(Ydesired153, dtype=np.float64)
H159 = np.asarray(Hpre118, dtype=np.float64).reshape(-1, 192)
P0_159 = np.asarray(load_original("core.c_out.weight"), dtype=np.float64).T @ R118

G159 = X159.T @ X159 / len(X159)
S159 = H159.T @ H159 / len(H159)
rhs159 = X159.T @ (Y159-X159 @ P0_159) / len(X159)
controllers159 = {}
routes159 = {}

print("=== FACTUAL-ANCHORED CONTROLLERS ===")
for penalty in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
    dp = np.linalg.solve(
        G159 + penalty*S159 + 1e-7*np.eye(192), rhs159)
    p = P0_159 + dp
    c = Cperp145 + R118 @ p.T
    u = np.linalg.solve(c, R118)
    controllers159[penalty] = p
    routes159[penalty] = u
    response = X159 @ p-Y159
    factual_delta = H159 @ dp
    pooled_delta = factual_delta.reshape(-1, 4, 4).mean(axis=1)
    print("penalty", penalty,
          "response RMSE", np.round(np.sqrt(np.mean(response**2, axis=0)), 5),
          "factual slot RMS", np.round(np.sqrt(np.mean(factual_delta**2, axis=0)), 5),
          "pooled RMS", np.round(np.sqrt(np.mean(pooled_delta**2, axis=0)), 5),
          "mean delta", np.round(factual_delta.mean(axis=0), 5),
          "write norm", round(float(np.linalg.norm(u)), 3),
          "route error", float(np.max(np.abs(c @ u-R118))))

with open("/work/notes.md", "a") as f159:
    f159.write(
        "\n## Run 159 / factual-anchored controller\n"
        "Recovery found controllers153 used centered factual covariance but had "
        "large unpenalized means (about 30 in one channel). They are not safe "
        "to submit. Refit private c_out correction relative to the original "
        "mapping using complete raw factual slot states. controllers159 and "
        "routes159 contain the new candidates. No additional weights saved.\n")
print("Edits remain:", edited())
```
