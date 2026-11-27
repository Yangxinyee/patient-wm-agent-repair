The cached states cover 3,935 bundle updates with a stay-separated holdout. I’ll fit a closed-form correction to the prelude’s final feed-forward layer, targeting the original slot outputs under the proposed routing. A substantial reduction in held-out reconstruction error would justify a factual test; poor reconstruction would reject this approach before adding treatment effects.

```python
import torch.nn.functional as F
import scipy.linalg as sla
torch.set_num_threads(8)

route220 = {k: np.array(v, copy=True)
            for k, v in np.load("/work/neutral204.npz").items()}
t220 = lambda a: torch.as_tensor(a, dtype=torch.float32)
orig220 = {k: t220(load_original(k)) for k in NAMES
           if k.startswith("core.prelude.0.")}

def att_features220(tokens, qw, kvw):
    x = F.layer_norm(tokens, (192,),
                     orig220["core.prelude.0.norm1.weight"],
                     orig220["core.prelude.0.norm1.bias"])
    b, n, _ = x.shape
    q = F.linear(x, t220(qw)).reshape(b, n, 4, 48).transpose(1, 2)
    k, v = F.linear(x, t220(kvw)).split(192, dim=-1)
    k = k.reshape(b, n, 4, 48).transpose(1, 2)
    v = v.reshape(b, n, 4, 48).transpose(1, 2)
    p = torch.softmax((q @ k.transpose(-1, -2)) / np.sqrt(48.), dim=-1)
    return (p @ v).transpose(1, 2).reshape(b, n, 192)[:, :4]

def ffn_features220(x):
    xn = F.layer_norm(x, (192,),
                     orig220["core.prelude.0.norm3.weight"],
                     orig220["core.prelude.0.norm3.bias"])
    return F.gelu(F.linear(xn, orig220["core.prelude.0.ffn.0.weight"],
                           orig220["core.prelude.0.ffn.0.bias"]))

route_z220 = F.linear(
    F.gelu(F.linear(input196, t220(route220["z_post.0.weight"]),
                    t220(route220["z_post.0.bias"]))),
    t220(route220["z_post.2.weight"]),
    t220(route220["z_post.2.bias"]))[:, :32]
route_token220 = F.linear(route_z220, t220(route220["z_in.weight"]),
                          t220(route220["z_in.bias"]))
tokens220 = old_tokens196.clone()
tokens220[:, 6] = route_token220[source201]

Xorig220, Xroute220 = [], []
for start220 in range(0, len(tokens220), 256):
    sl220 = slice(start220, start220 + 256)
    fo220 = att_features220(
        old_tokens196[sl220],
        orig220["core.prelude.0.attn.q.weight"],
        orig220["core.prelude.0.attn.kv.weight"])
    fr220 = att_features220(
        tokens220[sl220], route220["core.prelude.0.attn.q.weight"],
        route220["core.prelude.0.attn.kv.weight"])
    Xorig220.append(old_tokens196[sl220, :4] +
                    F.linear(fo220, orig220["core.prelude.0.attn.out.weight"]))
    Xroute220.append(tokens220[sl220, :4] +
                     F.linear(fr220, t220(route220["core.prelude.0.attn.out.weight"])))
Xorig220 = torch.cat(Xorig220).reshape(-1, 192)
Xroute220 = torch.cat(Xroute220).reshape(-1, 192)
Hroute220 = ffn_features220(Xroute220)
Yorig220 = Xorig220 + F.linear(ffn_features220(Xorig220),
                             orig220["core.prelude.0.ffn.2.weight"],
                             orig220["core.prelude.0.ffn.2.bias"])
Yroute220 = Xroute220 + F.linear(Hroute220,
                               orig220["core.prelude.0.ffn.2.weight"],
                               orig220["core.prelude.0.ffn.2.bias"])
residual220 = (Yorig220 - Yroute220).numpy().astype(np.float64)
A220 = Hroute220.numpy().astype(np.float64)
fit220 = np.repeat(np.asarray(train196)[source201], 4)
mu220 = A220[fit220].mean(0)
sd220 = np.maximum(A220[fit220].std(0), 0.05)
Ac220 = (A220 - mu220) / sd220
rmu220 = residual220[fit220].mean(0)
G220 = Ac220[fit220].T @ Ac220[fit220] / fit220.sum()
C220 = Ac220[fit220].T @ (residual220[fit220] - rmu220) / fit220.sum()
rms220 = lambda a: float(np.sqrt(np.mean(np.asarray(a)**2)))
print("Rows train / held:", int(fit220.sum()), int((~fit220).sum()))
print("Uncompensated final-slot RMS train / held:",
      rms220(residual220[fit220]), rms220(residual220[~fit220]))
print("Route donor head / channels:", private_head196,
      globals().get("channels204"), globals().get("head_start203"))
fits220 = []
for lam220 in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0):
    coef220 = sla.solve(G220 + lam220 * np.eye(768), C220,
                        assume_a="pos", check_finite=False)
    err220 = residual220 - (Ac220 @ coef220 + rmu220)
    score220 = rms220(err220[~fit220])
    fits220.append((score220, lam220, coef220))
    print("ridge", lam220, "RMS train", round(rms220(err220[fit220]), 6),
          "held", round(score220, 6))
best220 = min(fits220, key=lambda a: a[0])
deltaW220 = (best220[2] / sd220[:, None]).T
deltab220 = rmu220 - mu220 @ deltaW220.T
compensated220 = dict(route220)
compensated220["core.prelude.0.ffn.2.weight"] = (
    orig220["core.prelude.0.ffn.2.weight"].numpy() + deltaW220).astype(np.float32)
compensated220["core.prelude.0.ffn.2.bias"] = (
    orig220["core.prelude.0.ffn.2.bias"].numpy() + deltab220).astype(np.float32)
np.savez("/work/compensated220.npz", **compensated220)
print("Best ridge", best220[1], "held RMS", best220[0])
print("Correction weight / bias RMS:", rms220(deltaW220), rms220(deltab220))
print("Installed edits unchanged:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 220 — closed-form neutral-routing compensation\n"
            f"- Fitted only a final-FFN linear correction; no gradients or optimizer used.\n"
            f"- Uncompensated held slot RMS {rms220(residual220[~fit220]):.6g}; "
            f"best compensated held RMS {best220[0]:.6g}, ridge {best220[1]}.\n"
            "- Saved compensated220.npz but did not install it. GPU factual validation remains necessary.\n")
```