The existing recurrence almost erases the quiet state directions, so copying a dose signal into them would not be enough. I’ll fit a small linear read/write controller to preserve that signal, and check whether two attention value channels can be repurposed without changing the original attention output. Low factual error and accurate transmission are the acceptance tests.

```python
from scipy.linalg import null_space

# Quiet prelude directions, constrained against direct state-channel feedback.
quiet_basis125 = UE124[:, 1:35]
constraints125 = np.c_[gamma124[:, None]*R118, 1/gamma124]
null125 = null_space((quiet_basis125.T @ constraints125).T)
sub125 = quiet_basis125 @ null125
ev125, rot125 = np.linalg.eigh(sub125.T @ CE124 @ sub125)
QE125 = sub125 @ rot125[:, :4]
SP125 = QE125 / gamma124[:, None]
print("Quiet E RMS:", np.sqrt(np.mean((Ef124@QE125)**2, axis=0)))
print("State-feedback constraint residual:",
      np.linalg.norm(QE125.T@(gamma124[:, None]*R118)))

# Capture how incoming private state appears at the coda input.
capture125 = []
linear_before125 = linear63
def capture_pre125(x, prefix):
    if prefix == "core.c_out":
        capture125.append(x.detach().cpu().numpy().copy())
    return linear_backup118(x, prefix)

Jpre125 = []
try:
    linear63 = capture_pre125
    for j125 in range(4):
        pair125 = []
        for sign125 in (-1.0, 1.0):
            hh125 = hprev124 + sign125*torch.as_tensor(
                R118[:, j125], dtype=torch.float32)[None, None, :]
            oo125 = oprev124 + (
                norm63(hh125.mean(1), "core.out_norm")
                - norm63(hprev124.mean(1), "core.out_norm"))
            cc125, hf125 = context64(oo125, hh125, dt124)
            zz125 = mlp63(torch.cat(
                [cc125, e_valid60[ids124], u_valid60[ids124]], -1),
                "z_post")[:, :32]
            capture125.clear()
            transition64(hf125, e_valid60[ids124], u_valid60[ids124],
                         zz125, mem124, age124, mask124)
            pair125.append(capture125[-1].mean(1).astype(np.float64))
        Jpre125.append((pair125[1]-pair125[0])/2)
finally:
    linear63 = linear_before125
    capture125.clear()

Jpre125 = np.stack(Jpre125, axis=-1)  # bundles, features, incoming channels
Fflow125 = np.stack([
    R118.T @ (np.exp(-float(d)*flow124)[:, None]*R118)
    for d in dt124.numpy()])
Xstate125 = Jpre125.transpose(0, 2, 1).reshape(-1, 192)
Ystate125 = Fflow125.transpose(0, 2, 1).reshape(-1, 4)
Sstate125 = Xstate125.T @ Xstate125 / len(Xstate125)
rhs_state125 = Xstate125.T @ Ystate125 / len(Xstate125)
controllers125 = {}
for penalty125 in (0.1, 1.0, 10.0):
    P125 = np.linalg.solve(
        Sstate125 + penalty125*Cpre118 + np.eye(192)*1e-9,
        rhs_state125)
    controllers125[penalty125] = P125
    print("Controller", penalty125,
          "factual RMS", np.sqrt(np.mean((hp118@P125)**2, axis=0)).round(5),
          "response RMSE",
          np.sqrt(np.mean((Xstate125@P125-Ystate125)**2)).round(5),
          "norms", np.linalg.norm(P125, axis=0).round(3))

P125 = controllers125[1.0]
T125 = Jpre125.mean(0) @ np.linalg.inv(Fflow125.mean(0))
P125 = P125 @ np.linalg.inv(P125.T@T125).T
print("Controller identity:", np.round(P125.T@T125, 4))

# Recover old attention output after replacing two head-2 value channels.
Wkv125 = w63("core.prelude.0.attn.kv.weight").numpy().astype(np.float64).copy()
Wo125 = w63("core.prelude.0.attn.out.weight").numpy().astype(np.float64).copy()
V125 = Wkv125[192:]
raw125 = features116
pooled125 = np.concatenate([
    raw125[:, h] @ V125[h*48:(h+1)*48].T for h in range(4)], axis=1)
importance125 = pooled125[:, 96:144].std(0) * np.linalg.norm(Wo125[:, 96:144], axis=0)
chosen125 = np.argsort(importance125)[:2] + 96
Bdose125 = fits116[(2, 0.3)].copy()

# Exact constant functional of affine layer-normalized tokens.
gn125 = w63("core.prelude.0.norm1.weight").numpy().astype(np.float64)
bn125 = w63("core.prelude.0.norm1.bias").numpy().astype(np.float64)
constant125 = float(np.sum(bn125/gn125))
print("Layer-normalization constant:", constant125)
assert abs(constant125) > 1e-8
constvec125 = (1/gn125)/constant125
Vnew125 = V125.copy()
for col125, row125 in enumerate(chosen125):
    Vnew125[row125] = Bdose125[:-1, col125] + Bdose125[-1, col125]*constvec125

newpool125 = np.concatenate([
    raw125[:, h] @ Vnew125[h*48:(h+1)*48].T for h in range(4)], axis=1)
oldout125 = pooled125 @ Wo125.T
resid125 = oldout125 - newpool125@Wo125.T
reg125 = np.eye(192)*1e-7
repair125 = np.linalg.solve(
    newpool125.T@newpool125/len(newpool125)+reg125,
    newpool125.T@resid125/len(newpool125))
Wrepair125 = Wo125 + repair125.T
print("Replaced value rows:", chosen125)
print("Factual attention reconstruction RMS:",
      np.sqrt(np.mean((newpool125@Wrepair125.T-oldout125)**2)))
print("Reconstruction weight-change norm:", np.linalg.norm(Wrepair125-Wo125))
print("Dose-channel factual agreement RMS:",
      np.sqrt(np.mean((newpool125[:, chosen125] -
                       (np.c_[raw125[:, 2], np.ones(len(raw125))]@Bdose125))**2)))

for key125 in (("insulin", 2.0), ("insulin", 3.0),
               ("norepinephrine", 2.0), ("norepinephrine", 0.0)):
    mask125 = np.array([tuple(g) == key125 for g in groups116])
    dr125 = D116[mask125]
    oldd125 = np.concatenate([
        dr125[:, h] @ V125[h*48:(h+1)*48].T for h in range(4)], axis=1)
    newd125 = np.concatenate([
        dr125[:, h] @ Vnew125[h*48:(h+1)*48].T for h in range(4)], axis=1)
    print("Counterfactual reconstruction", key125,
          np.sqrt(np.mean((newd125@Wrepair125.T-oldd125@Wo125.T)**2)).round(6))
print("Current edits:", edited())

with open("/work/notes.md", "a") as f125:
    f125.write("\n## Run 125 / private controller and attention wire\n"
               "Existing private-state transition gains are near zero. "
               "Fitted a covariance-penalized controller from finite state "
               "perturbations and an attention-output reconstruction after "
               "replacing two head-2 value channels. Variables P125/T125, "
               "QE125/SP125, Vnew125/Wrepair125/chosen125 hold the proposed "
               "analytical components. No weights saved.\n")
```