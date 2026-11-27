The earlier posterior edit was largely ignored because attention almost never selected the latent token. I will test whether one attention head can be repurposed to carry an explicit dose signal while the other three preserve its original contribution. Small held-out reconstruction errors would justify a factual-only test with the new signal set to zero.

```python
import torch.nn.functional as F

def wt145(name):
    return torch.as_tensor(load_original(name), dtype=torch.float32)

def ln145(x, prefix):
    return F.layer_norm(x, (192,), wt145(prefix + ".weight"), wt145(prefix + ".bias"))

def prelude_parts145(tokens):
    x = torch.as_tensor(tokens, dtype=torch.float32)
    p = "core.prelude.0"
    n = ln145(x, p + ".norm1")
    q = F.linear(n, wt145(p + ".attn.q.weight")).reshape(-1, 7, 4, 48).transpose(1, 2)
    k, v = F.linear(n, wt145(p + ".attn.kv.weight")).chunk(2, -1)
    k = k.reshape(-1, 7, 4, 48).transpose(1, 2)
    v = v.reshape(-1, 7, 4, 48).transpose(1, 2)
    attention = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(48), -1)
    pooled = (attention @ v).transpose(1, 2).reshape(-1, 7, 192)
    attout = F.linear(pooled, wt145(p + ".attn.out.weight"))
    after = x + attout
    hidden = F.gelu(F.linear(ln145(after, p + ".norm3"),
                            wt145(p + ".ffn.0.weight"), wt145(p + ".ffn.0.bias")))
    raw = after + F.linear(hidden, wt145(p + ".ffn.2.weight"),
                           wt145(p + ".ffn.2.bias"))
    E = ln145(raw[:, :4], "core.prelude_norm")
    return n, pooled, attout, after, hidden, raw, E, attention

parts145 = prelude_parts145(tokens81)
print("Prelude replay RMSE:", float(torch.mean(
    (parts145[6] - torch.as_tensor(E81))**2).sqrt()))
print("Attention replay RMSE:", float(torch.mean(
    (parts145[7] - torch.as_tensor(attention81))**2).sqrt()))

norm145 = parts145[0].numpy()
Xgate145 = norm145.reshape(-1, 192).astype(np.float64)
ygate145 = np.tile([0., 0., 0., 0., 0., 0., 1.], len(norm145))
gate145 = np.linalg.solve(
    Xgate145.T @ Xgate145 + 1e-6 * np.eye(192),
    Xgate145.T @ ygate145
)
pgate145 = (Xgate145 @ gate145).reshape(-1, 7)
print("Latent gate: ordinary SD, latent min/max:",
      pgate145[:, :6].std(), pgate145[:, 6].min(), pgate145[:, 6].max())

gamma145 = wt145("core.prelude.0.norm1.weight").numpy().astype(np.float64)
beta145 = wt145("core.prelude.0.norm1.bias").numpy().astype(np.float64)
constant145 = (1. / gamma145) / np.sum(beta145 / gamma145)
print("Exact constant projection max error:",
      np.max(np.abs(Xgate145 @ constant145 - 1.)))

P145 = parts145[1][:, :4].numpy().reshape(-1, 192).astype(np.float64)
O145 = parts145[2][:, :4].numpy().reshape(-1, 192).astype(np.float64)
stay145 = np.repeat(np.asarray(stay88), 4)
unique145 = np.unique(stay145)
train145 = np.isin(stay145, unique145[:80])
valid145 = ~train145
headfits145 = {}
for head145 in range(4):
    cols145 = np.array([j for j in range(192) if j // 48 != head145])
    X145 = np.column_stack((P145[:, cols145], np.ones(len(P145))))
    Wbase145 = wt145("core.prelude.0.attn.out.weight").numpy()[:, cols145].T
    base145 = np.vstack((Wbase145, np.zeros((1, 192))))
    residual145 = O145 - X145 @ base145
    C145 = X145[train145].T @ X145[train145]
    penalty145 = 1e-5 * np.trace(C145[:-1, :-1]) / 144
    correction145 = np.linalg.solve(C145 + penalty145 * np.eye(145),
                                    X145[train145].T @ residual145[train145])
    fit145 = base145 + correction145
    pred145 = X145 @ fit145
    err145 = np.sqrt(np.mean((pred145[valid145] - O145[valid145])**2))
    headfits145[head145] = (cols145, fit145, err145)
    print("Head", head145, "held-out attention-output RMSE", err145,
          "original output RMS", np.sqrt(np.mean(O145[valid145]**2)))

print("Action decoder activation check")
u145 = torch.as_tensor(u81[:100], dtype=torch.float32)
aW145 = torch.as_tensor(action_W105, dtype=torch.float32)
ab145 = torch.as_tensor(action_b105, dtype=torch.float32)
rawa145 = F.linear(u145, aW145, ab145)
for label145, a145 in (("gelu", F.gelu(rawa145)),
                       ("relu", F.relu(rawa145)),
                       ("tanh", torch.tanh(rawa145))):
    print(label145, "feature range", float(a145.min()), float(a145.max()))
print("Decoder coefficients shape:", np.shape(action_coef105))
print("Current edits:", edited())

with open("/work/notes.md", "a") as f145:
    f145.write("\nPython run 145; forwards 9. Testing a dedicated latent-transport attention head, with zero payload first. Replayed original prelude and fitted three-head compensation on factual training states with whole-stay holdout; no weights changed.\n")
```