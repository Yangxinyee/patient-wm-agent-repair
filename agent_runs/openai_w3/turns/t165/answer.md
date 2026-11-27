The zero-payload channel passes the factual test: NLL changed by only about 0.00003 nats. Next I will check the retained dose decoder across all dose factors and measure how accurately context predicts the recorded dose. Small factual dose residuals and reliable counterfactual decoding would make a selective correction feasible.

```python
print("Retained action-case arrays")
for name148 in sorted(list(globals())):
    obj148 = globals()[name148]
    if isinstance(obj148, (np.ndarray, torch.Tensor)):
        shape148 = tuple(obj148.shape)
        if len(shape148) == 2 and shape148[0] == 46343 and shape148[1] in (2, 4, 128, 129, 192):
            print(name148, shape148)
print("Case/target metadata")
for name148 in sorted(list(globals())):
    if name148.endswith(("104", "105")) and any(
        s in name148.lower() for s in ("label", "target", "factor", "case", "dose", "rate", "policy")
    ):
        obj148 = globals()[name148]
        print(name148, getattr(obj148, "shape", type(obj148)),
              repr(obj148)[:160] if not isinstance(obj148, (np.ndarray, torch.Tensor)) else "")

rawa148 = F.linear(torch.as_tensor(u81[:100], dtype=torch.float32),
                   torch.as_tensor(action_W105, dtype=torch.float32),
                   torch.as_tensor(action_b105, dtype=torch.float32))
ha148 = np.asarray(HA105)[:100, :128]
for label148, aa148 in (
    ("gelu", F.gelu(rawa148)), ("relu", F.relu(rawa148)),
    ("tanh", torch.tanh(rawa148)), ("silu", F.silu(rawa148))
):
    print("HA105 activation match", label148,
          np.sqrt(np.mean((aa148.numpy()-ha148)**2)))
print("HA105 final column:", np.min(HA105[:, -1]), np.max(HA105[:, -1]))
print("Original decoder RMSE, first two columns:",
      np.sqrt(np.mean((np.asarray(action_pred105)[:4213, :2]-np.asarray(rate88))**2, axis=0)))
print("Original decoder four-column ranges:",
      np.min(action_pred105[:4213], axis=0), np.max(action_pred105[:4213], axis=0))

if "pilot_weights_53" in globals():
    print("Pilot posterior modified hidden rows:")
    pp148 = pilot_weights_53
    pW148 = torch.as_tensor(pp148["z_post.0.weight"])
    change148 = torch.linalg.vector_norm(
        pW148-wt145("z_post.0.weight"), dim=1).numpy()
    rows148 = np.flatnonzero(change148 > 1e-4)
    print(rows148)
    print("Their context/action coefficient norms:",
          torch.linalg.vector_norm(pW148[rows148, :192], dim=1).numpy(),
          torch.linalg.vector_norm(pW148[rows148, 384:], dim=1).numpy())

Cctx148 = np.asarray(ctx81, dtype=np.float64)
stctx148 = np.asarray(stay88)
trctx148 = np.isin(stctx148, np.unique(stctx148)[:80])
policyfits148 = {}
for j148 in range(2):
    active148 = np.asarray(rate88)[:, j148] > 0
    tr148 = trctx148 & active148
    va148 = ~trctx148 & active148
    xm148 = Cctx148[tr148].mean(0)
    xs148 = Cctx148[tr148].std(0) + 1e-3
    X148 = (Cctx148-xm148)/xs148
    ym148 = np.asarray(rate88)[tr148, j148].mean()
    CV148 = X148[tr148].T @ X148[tr148]
    RV148 = X148[tr148].T @ (np.asarray(rate88)[tr148, j148]-ym148)
    for ridge148 in (0.01, 0.1, 1., 10., 100.):
        b148 = np.linalg.solve(CV148 + ridge148*np.eye(192), RV148)
        pred148 = X148 @ b148 + ym148
        err148 = np.sqrt(np.mean((pred148[va148]-np.asarray(rate88)[va148, j148])**2))
        policyfits148[(j148, ridge148)] = (
            b148/xs148, ym148-xm148 @ (b148/xs148), err148
        )
    best148 = min((k for k in policyfits148 if k[0] == j148),
                  key=lambda k: policyfits148[k][2])
    print("Policy", j148, "best ridge / held-out RMSE / rate mean, SD:",
          best148[1], policyfits148[best148][2],
          np.asarray(rate88)[active148, j148].mean(),
          np.asarray(rate88)[active148, j148].std())

zin148 = wt145("z_in.weight").numpy().astype(np.float64)
zb148 = wt145("z_in.bias").numpy().astype(np.float64)
print("Latent raw SD mean/range:",
      rawtokens146[:, 6].std(dim=-1).mean().item(),
      rawtokens146[:, 6].std(dim=-1).min().item(),
      rawtokens146[:, 6].std(dim=-1).max().item())
print("Current edit count:", len(edited()))

with open("/work/notes.md", "a") as f148:
    f148.write("\nPython run 148; forwards 10. Zero-payload latent-head transport passed tuning factual test (1.63982 original, 1.63985 edited). Checking retained nonlinear action decoder and context dose-policy accuracy before adding a causal signal.\n")
```