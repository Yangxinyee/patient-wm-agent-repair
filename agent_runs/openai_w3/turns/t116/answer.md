This repeats the previous output, so no new calibration result is available and no weights have changed. I’ll run the finite-difference calibration next. Agreement with the original recurrent outputs, followed by correctly signed glucose and MAP sensitivities, would justify testing the first effective repair.

```python
import numpy as np
import torch

# Refine the dose decoder, retaining only held-out improvements.
pen105 = 1e-13*np.eye(193)
pen105[-1, -1] = 1e-15
ld105 = np.linalg.solve(
    XX103[tr103].T @ XX103[tr103]/tr103.sum() + pen105,
    XX103[tr103].T @ YY103[tr103]/tr103.sum())
print("LOW-RIDGE held RMSE",
      np.sqrt(np.mean((XX103[~tr103] @ ld105-YY103[~tr103])**2, axis=0)))
action_W105 = action_W104.copy()
action_b105 = action_b104.copy()
action_W105[:4], action_W105[4:8] = ld105[:192].T, -ld105[:192].T
action_b105[:4], action_b105[4:8] = ld105[192], -ld105[192]
HA105 = torch.nn.functional.gelu(torch.as_tensor(
    U103 @ action_W105.T + action_b105, dtype=torch.float32)).numpy().astype(np.float64)
HA105 = np.column_stack([HA105, np.ones(len(HA105))])
action_fits105 = {}
for lam in [1e-8, 1e-6, 1e-4]:
    penalty = lam*np.eye(129)
    penalty[-1, -1] = 1e-12
    b = np.linalg.solve(
        HA105[tr103].T @ HA105[tr103]/tr103.sum() + penalty,
        HA105[tr103].T @ YY103[tr103]/tr103.sum())
    action_fits105[lam] = b
    print("refined GELU", lam, "held dose RMSE",
          np.sqrt(np.mean((HA105[~tr103] @ b-YY103[~tr103])[:, :2]**2, axis=0)))
best_action_key105 = min(action_fits105, key=lambda k:
    np.mean(((HA105[~tr103] @ action_fits105[k]-YY103[~tr103])[:, :2] /
             np.array([2., .4]))**2))
action_coef105 = action_fits105[best_action_key105]
action_pred105 = HA105 @ action_coef105

# Recover saved attribute inputs directly instead of guessing numeric-field order.
attr_static105 = torch.as_tensor(a197)-torch.as_tensor(q97)
centres105 = torch.cat([torch.zeros((200, 1)), torch.as_tensor(centres97)], dim=1)
def means105(view):
    q = layer_norm(view + W["type_emb.weight"][1], "query_norm")
    logits = mlp(q + attr_static105, "attr_heads.value_bin") + torch.as_tensor(transition97)
    return (torch.softmax(logits, dim=-1)*centres105).sum(dim=1)
base_view105 = torch.as_tensor(view97)
print("RECOVERED READOUT RMSE",
      np.sqrt(np.mean((means105(base_view105).numpy()-observed_mean97)**2)))

read_codes105 = np.array([5, 6, 7, 10])
G105 = np.zeros((4, 192))
eps105 = .004
for k in range(192):
    step = torch.zeros_like(base_view105)
    step[:, k] = eps105
    derivative = ((means105(base_view105+step)-means105(base_view105-step)) /
                  (2*eps105)).numpy()
    for j, code in enumerate(read_codes105):
        G105[j, k] = derivative[np.asarray(codes97) == code].mean()

def parts105(h, e, u, z):
    tokens = torch.cat([h+W["core.slots"][None], e[:, None], u[:, None],
                        linear(z, "z_in")[:, None]], dim=1)
    normed = layer_norm(tokens, "core.prelude.0.norm1")
    q = linear(normed, "core.prelude.0.attn.q").reshape(-1, 7, 4, 48).transpose(1, 2)
    k, v = linear(normed, "core.prelude.0.attn.kv").chunk(2, dim=-1)
    k = k.reshape(-1, 7, 4, 48).transpose(1, 2)
    v = v.reshape(-1, 7, 4, 48).transpose(1, 2)
    weights = torch.softmax(q @ k.transpose(-1, -2)/np.sqrt(48), dim=-1)
    pooled = (weights @ v).transpose(1, 2).reshape(-1, 7, 192)
    x1 = tokens + linear(pooled, "core.prelude.0.attn.out")
    x2 = x1 + mlp(layer_norm(x1, "core.prelude.0.norm3"), "core.prelude.0.ffn")
    return x1[:, :4], layer_norm(x2[:, :4], "core.prelude_norm"), weights, pooled

pick105 = np.unique(np.concatenate([
    np.flatnonzero(rate88[:, 0] > 0)[::max(1, np.sum(rate88[:, 0]>0)//8)][:8],
    np.flatnonzero(rate88[:, 1] > 0)[::max(1, np.sum(rate88[:, 1]>0)//8)][:8]
]))
abs105 = np.asarray(idx81)[pick105]
hp105 = h81[pick105]
xp105, Ep105, ap105, vp105 = parts105(
    hp105, base_e103[pick105], base_u103[pick105], base_z103[pick105])
print("PRELUDE REPLAY RMSE",
      torch.sqrt(torch.mean((Ep105-E81[pick105])**2)).item())
memory105, mask105, age105 = memory82(abs105, True)
nh105, no105 = state_update82(hp105, Ep105, memory105, mask105, age105)
print("CORE REPLAY out RMSE",
      np.sqrt(np.mean((no105.numpy()-S["x"][abs105, :192])**2)))

next_dt105 = torch.as_tensor(
    (np.floor(S["hour"][abs105])+1-S["hour"][abs105])*3600,
    dtype=torch.float32)
def next_view105(h, out):
    hf = h*torch.exp(-next_dt105[:, None, None]*torch.exp(W["core.flow.a"])[None, None])
    ctx = out + layer_norm(hf.mean(dim=1), "core.out_norm")
    ctx = ctx + mlp(torch.log(next_dt105.clamp_min(1))[:, None]/10, "gap")
    return ctx + linear(mlp(ctx, "z_prior")[:, :32], "z_head")

out_map105 = W["core.prelude.0.attn.out.weight"][:, 96:144]
Jview105 = np.zeros((192, 48))
for k in range(48):
    step = out_map105[:, k][None, None]*eps105
    views = []
    for sign in [1., -1.]:
        xx = xp105 + sign*step
        EE = layer_norm(
            xx + mlp(layer_norm(xx, "core.prelude.0.norm3"), "core.prelude.0.ffn"),
            "core.prelude_norm")
        hh, oo = state_update82(hp105, EE, memory105, mask105, age105)
        views.append(next_view105(hh, oo))
    Jview105[:, k] = ((views[0]-views[1])/(2*eps105)).mean(dim=0).numpy()

Jvalue105 = G105 @ Jview105
target105 = np.array([[0., 3.2], [-1.94, 0.], [0., 3.2], [-1.94, 0.]])
ridge105 = 1e-4*np.trace(Jvalue105 @ Jvalue105.T)/4
value_directions105 = Jvalue105.T @ np.linalg.solve(
    Jvalue105 @ Jvalue105.T + ridge105*np.eye(4), target105)
print("CALIBRATED LOCAL SHIFTS (cuff, finger, arterial, lab)\n",
      Jvalue105 @ value_directions105)
print("VALUE-CHANNEL NORMS", np.linalg.norm(value_directions105, axis=0))
print("PRELUDE INJECTION NORMS",
      np.linalg.norm(out_map105.numpy() @ value_directions105, axis=0))
print("CURRENT EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 105; forwards 7. Refined explicit dose decoder and calibrated "
            "head-2 value-channel directions with numerical finite differences; exact "
            "prelude/core replay checked. No model gradients or weight edits.\n")
```
