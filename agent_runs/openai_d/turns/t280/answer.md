The four changed neurons form separate insulin and NE pairs, and the caches provide an exact value-head check. I’ll fit training-only regressions that cancel the NE pair’s factual contribution while constraining their approximate NE sensitivity. Lower held-out NLL would justify a rollout test; poor generalization would refute this approach.

```python
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy.linalg import null_space

r240_ne_rows = (48, 171)
r240_keep = np.setdiff1d(np.arange(192), r239_rows)
r240_q = r239_safe[r239_keys[2]][:, 48].astype(np.float64)
r240_probe = np.asarray(r177_probe_direction, dtype=np.float64).reshape(192)
r240_ids = np.sort(c210_train["meta"]["stay_id"].unique())
r240_fit = c210_train["meta"]["stay_id"].isin(r240_ids[:60]).to_numpy()
r240_val = ~r240_fit

def r240_prepare(c):
    X, H = c["X"], c["H0"]
    w0 = torch.from_numpy(r239_safe[r239_keys[0]][r239_rows])
    b0 = torch.from_numpy(r239_safe[r239_keys[1]][r239_rows])
    hc = F.gelu(F.linear(X, w0, b0))
    dw2 = torch.from_numpy(r239_safe[r239_keys[2]] - r239_orig[r239_keys[2]])
    db2 = torch.from_numpy(r239_safe[r239_keys[3]] - r239_orig[r239_keys[3]])
    L = c["L0"] + F.linear(H, dw2, db2)
    L = L + F.linear(hc - H[:, :, r239_rows],
                     torch.from_numpy(r239_safe[r239_keys[2]][:, r239_rows]))
    j48 = int(np.flatnonzero(r239_rows == 48)[0])
    j171 = int(np.flatnonzero(r239_rows == 171)[0])
    return {"L": L, "R": hc[:, :, j48] - hc[:, :, j171]}

r240_pre = {name: r240_prepare(c) for name, c in
            (("train", c210_train), ("tune", c210_tune))}
r240_Hfit = c210_train["H0"][::4, r240_fit][:, :, r240_keep].reshape(
    -1, len(r240_keep)).double().numpy()
r240_Yfit = r240_pre["train"]["R"][::4, r240_fit].reshape(-1).double().numpy()
r240_mu = r240_Hfit.mean(axis=0)
r240_sd = np.maximum(r240_Hfit.std(axis=0), 0.02)
r240_Z = (r240_Hfit - r240_mu) / r240_sd
r240_ymu = float(r240_Yfit.mean())
r240_C = r240_Z.T @ r240_Z / len(r240_Z)
r240_cross = r240_Z.T @ (r240_Yfit - r240_ymu) / len(r240_Z)

# Local sensitivity of ORIGINAL retained hidden features to the old NE probe.
# This is only a proxy constraint, not evidence about real dose-edit rollouts.
r240_Xd = c210_train["X"][::8, r240_fit].reshape(-1, 192)
r240_wkeep = torch.from_numpy(r239_orig[r239_keys[0]][r240_keep])
r240_bkeep = torch.from_numpy(r239_orig[r239_keys[1]][r240_keep])
r240_vp = torch.from_numpy(r240_probe.astype(np.float32))
r240_eps = 0.10
r240_D = ((F.gelu(F.linear(r240_Xd + r240_eps*r240_vp, r240_wkeep, r240_bkeep))
           - F.gelu(F.linear(r240_Xd - r240_eps*r240_vp, r240_wkeep, r240_bkeep)))
          / (2*r240_eps)).double().numpy() / r240_sd
r240_ev = np.tile(c210_train["meta"].loc[r240_fit, "event"].to_numpy(), 4)
r240_dmeans = np.stack([r240_D[r240_ev == e].mean(axis=0)
                       for e in np.unique(r240_ev)])
r240_evals, r240_evecs = np.linalg.eigh(r240_D.T @ r240_D / len(r240_D))

def r240_loss(c, pre, scalar):
    logits = pre["L"] + scalar[:, :, None] * torch.from_numpy(
        r240_q.astype(np.float32))[None, None, :]
    p = torch.softmax(logits, dim=-1).mean(dim=0)
    return (-torch.log(p[torch.arange(len(c["y"])), c["y"]].clamp_min(1e-30))).numpy()

r240_baselines = {}
for label, c in (("train", c210_train), ("tune", c210_tune)):
    loss = r240_loss(c, r240_pre[label], torch.zeros_like(r240_pre[label]["R"]))
    r240_baselines[label] = loss
print("Safe cached deltas:",
      "train", float(np.mean(r240_baselines["train"]-c210_train["loss0"])),
      "tune100", float(np.mean(r240_baselines["tune"]-c210_tune["loss0"])))

r240_regressions, r240_results = {}, []
for npc in (-1, 0, 4, 8, 16):
    if npc == -1:
        B = np.eye(len(r240_keep))
    else:
        constraints = r240_dmeans
        if npc:
            constraints = np.vstack([constraints, r240_evecs[:, -npc:].T])
        B = null_space(constraints, rcond=1e-7)
    for ridge in (0.001, 0.01, 0.1):
        coef = B @ np.linalg.solve(
            B.T @ r240_C @ B + ridge*np.eye(B.shape[1]),
            B.T @ r240_cross)
        beta = np.zeros(192, dtype=np.float64)
        beta[r240_keep] = coef/r240_sd
        intercept = r240_ymu - float(r240_mu @ beta[r240_keep])
        pred = {}
        for label, c in (("train", c210_train), ("tune", c210_tune)):
            pred[label] = F.linear(c["H0"],
                torch.from_numpy(beta.astype(np.float32))[None, :],
                torch.tensor([intercept], dtype=torch.float32)).squeeze(-1)
        key = (npc, ridge)
        r240_regressions[key] = {"beta": beta, "intercept": intercept,
                                "pred": pred, "null_dim": B.shape[1]}
        for amp in (0.6, 0.8, 1.0):
            factor = amp/0.4
            for frac in (0.5, 1.0):
                losses = {}
                for label, c in (("train", c210_train), ("tune", c210_tune)):
                    scalar = ((factor-1)*r240_pre[label]["R"]
                              - factor*frac*pred[label])
                    losses[label] = r240_loss(c, r240_pre[label], scalar)-c["loss0"]
                r240_results.append({
                    "npc": npc, "ridge": ridge, "ne_amp": amp, "cancel": frac,
                    "train_fit": float(losses["train"][r240_fit].mean()),
                    "train_validation": float(losses["train"][r240_val].mean()),
                    "tune100": float(losses["tune"].mean()),
                    "probe_mean_max": float(np.max(np.abs(r240_dmeans @ coef))),
                    "null_dim": B.shape[1]})
r240_table = pd.DataFrame(r240_results)
print("\n=== BEST PROXY-PROTECTED CANCELLATION SCREENS ===")
print(r240_table.query("npc >= 0").sort_values(
    ["tune100", "train_validation"]).head(24).to_string(index=False))
print("\n=== UNCONSTRAINED DIAGNOSTIC ===")
print(r240_table.query("npc == -1").sort_values("tune100").head(6).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 240: training-only nonlinear factual cancellation screen\n"
            "Fitted ridge predictions of the installed NE pair's factual activation "
            "using 188 unchanged original hidden features, on first 60 training stays. "
            "Protected event-average and optional principal-component sensitivity to "
            "the approximate raw NE probe. Screened NE amplitude .6/.8/1.0 and partial/full "
            "cancellation on training-held-out and tuning caches. No weights installed. "
            "Proxy constraints do not establish real dose-edit preservation.\n")
    f.write(r240_table.query("npc >= 0").sort_values(
        ["tune100", "train_validation"]).head(12).to_string(index=False) + "\n")
```