The latent token is very cleanly distinguishable from ordinary tokens, which makes an isolated channel plausible. However, the linear dose decoder is inaccurate, especially for norepinephrine. I’ll improve that decoder with a small closed-form feature regression and check a low-leakage payload direction. Held-out proportional responses across all dose factors would support proceeding to an edit.

```python
import numpy as np
import torch
from scipy.linalg import null_space
from sklearn.svm import LinearSVC

print("LESS-REGULARIZED LINEAR DECODERS")
linear_decoders104 = {}
for lam in [1e-7, 1e-9, 1e-11]:
    penalty = lam*np.eye(193)
    penalty[-1, -1] = 1e-12
    b = np.linalg.solve(XX103[tr103].T @ XX103[tr103]/tr103.sum() + penalty,
                        XX103[tr103].T @ YY103[tr103]/tr103.sum())
    linear_decoders104[lam] = b
    pp = XX103 @ b
    print(lam, "held RMSE", np.round(
        np.sqrt(np.mean((pp[~tr103]-YY103[~tr103])**2, axis=0)), 4))

# An own statistical model: fixed GELU features, ridge-fitted outputs.
um104 = U103[tr103].mean(axis=0)
uc104 = np.cov(U103[tr103], rowvar=False)
ue104, uv104 = np.linalg.eigh(uc104)
pc104 = uv104[:, -24:] / np.sqrt(np.maximum(ue104[-24:], 1e-4))[None]
rng104 = np.random.default_rng(104)
ld104 = linear_decoders104[1e-9]
expert104 = ld104[:192].T
expertb104 = ld104[192]
random104 = rng104.normal(size=(72, 24))/np.sqrt(24)
action_W104 = np.vstack([expert104, -expert104,
                         pc104.T, -pc104.T, random104 @ pc104.T]).astype(np.float32)
action_b104 = np.concatenate([
    expertb104, -expertb104,
    -pc104.T @ um104, pc104.T @ um104,
    -(random104 @ pc104.T) @ um104 + rng104.uniform(-1.5, 1.5, 72)
]).astype(np.float32)
HA104 = torch.nn.functional.gelu(torch.as_tensor(
    U103 @ action_W104.T + action_b104, dtype=torch.float32)).numpy().astype(np.float64)
HA104 = np.column_stack([HA104, np.ones(len(HA104))])
action_fits104 = {}
for lam in [1e-7, 1e-5, 1e-3]:
    penalty = lam*np.eye(129)
    penalty[-1, -1] = 1e-12
    b = np.linalg.solve(HA104[tr103].T @ HA104[tr103]/tr103.sum() + penalty,
                        HA104[tr103].T @ YY103[tr103]/tr103.sum())
    pp = HA104 @ b
    action_fits104[lam] = b
    print("GELU", lam, "held RMSE", np.round(
        np.sqrt(np.mean((pp[~tr103]-YY103[~tr103])**2, axis=0)), 4))

best_action_key104 = min(action_fits104, key=lambda k:
    np.mean(((HA104[~tr103] @ action_fits104[k] - YY103[~tr103])[:, :2] /
             np.array([2., .4]))**2))
action_coef104 = action_fits104[best_action_key104]
action_pred104 = HA104 @ action_coef104
n104 = len(rate88)
print("\nCHOSEN ACTION DECODER", best_action_key104)
for si, factors in enumerate(scenarios103[1:], 1):
    j = 0 if factors[0] != 1 else 1
    active = (~tr_bundle103) & (rate88[:, j] > 0)
    delta = action_pred104[si*n104:(si+1)*n104, j] - action_pred104[:n104, j]
    target = rate88[:, j]*(factors[j]-1)
    print(["insulin", "norepinephrine"][j], factors[j],
          "held mean target/decoded",
          np.round([target[active].mean(), delta[active].mean()], 4))

print("\nDRUG-PRESENCE MARGINS")
presence104 = []
for j in range(2):
    clf = LinearSVC(C=100., dual=False, max_iter=5000, tol=1e-6)
    clf.fit(U103[tr103], G103[tr103, j].astype(int))
    margin = clf.decision_function(U103[~tr103])
    y = G103[~tr103, j] > .5
    print(j, "held classification error", np.mean((margin > 0) != y),
          "signed margin quantiles",
          np.round(np.quantile(margin*(2*y-1), [.001,.01,.5]), 3))
    presence104.append(clf)

# Select payload features with little factual or direct-dose leakage.
pool104 = PF89[:, 2].astype(np.float64)
pool_mu104 = pool104.mean(axis=0)
pool_cov104 = np.cov(pool104, rowvar=False)
pool_delta104 = PD89[:, 2].astype(np.float64)
payload_cov104 = pool_cov104 + .1*(pool_delta104.T @ pool_delta104/len(pool_delta104))
zraw104 = linear(base_z103, "z_in").numpy().mean(axis=0)
zbase_norm104 = layer_norm(torch.as_tensor(zraw104)[None],
                          "core.prelude.0.norm1").numpy()[0].astype(np.float64)
gamma104 = W["core.prelude.0.norm1.weight"].numpy().astype(np.float64)
constraints104 = np.vstack([1/gamma104, zbase_norm104-pool_mu104, gate103])
null104 = null_space(constraints104)
pe104, pv104 = np.linalg.eigh(null104.T @ payload_cov104 @ null104)
payload104 = null104 @ pv104[:, :2]
print("\nPAYLOAD FEATURE SD, factual / direct dose",
      np.std((pool104-pool_mu104) @ payload104, axis=0),
      np.sqrt(np.mean((pool_delta104 @ payload104)**2, axis=0)))
print("Baseline latent payload",
      (zbase_norm104-pool_mu104) @ payload104)

# Recover the best saved value-head convention for finite-difference calibration.
def means104(view):
    q = layer_norm(view + W["type_emb.weight"][1], "query_norm")
    attrs = q + torch.as_tensor(base97) + W["last_numeric.weight"][:, 6][None]
    attrs = attrs + torch.as_tensor(zval97)[:, None]*W["last_numeric.weight"][:, 7][None]
    logits = mlp(attrs, "attr_heads.value_bin") + torch.as_tensor(transition97)
    p = torch.softmax(logits, dim=-1)
    centres = torch.cat([torch.zeros((len(p), 1)), torch.as_tensor(centres97)], dim=1)
    return (p*centres).sum(dim=1)

means_replay104 = means104(torch.as_tensor(view97)).numpy()
print("\nVALUE-HEAD REPLAY RMSE",
      np.sqrt(np.mean((means_replay104-observed_mean97)**2)))
print("Codes in replay", np.unique(codes97, return_counts=True))
print("\nCURRENT EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write("\nPython runs 103-104; forwards 7. Latent-token gate is strongly isolated: "
            "ordinary-token SD .00092, changed-action SD .00064 versus latent mean 1. "
            "Testing explicit dose decoder and low-variance attention-payload features "
            "for a posterior innovation channel. No weights edited yet.\n")
```
