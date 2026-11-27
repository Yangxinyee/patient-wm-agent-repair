The discrepancy was a cached-token convention: rebuilding the raw tokens matches the verified prelude essentially exactly. I will now repeat the head-compensation test on those inputs and prepare a zero-payload edit. Low held-out slot error would support a factual forward test; a large error would rule this design out.

```python
parts147 = rawparts146
norm147 = parts147[0].numpy().astype(np.float64)
Xgate147 = norm147.reshape(-1, 192)
ygate147 = np.tile([0., 0., 0., 0., 0., 0., 1.], len(norm147))
gate147 = np.linalg.solve(Xgate147.T @ Xgate147 + 1e-6*np.eye(192),
                          Xgate147.T @ ygate147)
pgate147 = (Xgate147 @ gate147).reshape(-1, 7)
print("Gate ordinary SD/max; latent min/max:",
      pgate147[:, :6].std(), np.max(np.abs(pgate147[:, :6])),
      pgate147[:, 6].min(), pgate147[:, 6].max())

P147 = parts147[1][:, :4].numpy().reshape(-1, 192).astype(np.float64)
O147 = parts147[2][:, :4].numpy().reshape(-1, 192).astype(np.float64)
R147 = parts147[5][:, :4].numpy().reshape(-1, 192)
rawslot147 = rawtokens146[:, :4].reshape(-1, 192)
headfits147 = {}

for head147 in range(4):
    cols147 = np.array([j for j in range(192) if j//48 != head147])
    X147 = np.column_stack((P147[:, cols147], np.ones(len(P147))))
    base147 = np.vstack((
        wt145("core.prelude.0.attn.out.weight").numpy()[:, cols147].T,
        np.zeros((1, 192))
    ))
    residual147 = O147 - X147 @ base147
    C147 = X147[train145].T @ X147[train145]
    penalty147 = 1e-5*np.trace(C147[:-1, :-1])/144
    fit147 = base147 + np.linalg.solve(
        C147 + penalty147*np.eye(145),
        X147[train145].T @ residual147[train145]
    )
    pred147 = X147 @ fit147
    after147 = rawslot147 + torch.as_tensor(pred147, dtype=torch.float32)
    hidden147 = F.gelu(F.linear(
        ln145(after147, "core.prelude.0.norm3"),
        wt145("core.prelude.0.ffn.0.weight"),
        wt145("core.prelude.0.ffn.0.bias")))
    raw147 = after147 + F.linear(
        hidden147, wt145("core.prelude.0.ffn.2.weight"),
        wt145("core.prelude.0.ffn.2.bias"))
    enew147 = ln145(raw147, "core.prelude_norm").numpy()
    eorig147 = np.asarray(E81).reshape(-1, 192)
    eerr147 = np.sqrt(np.mean((enew147[valid145]-eorig147[valid145])**2))
    aerr147 = np.sqrt(np.mean((pred147[valid145]-O147[valid145])**2))
    headfits147[head147] = (cols147, fit147, eerr147, hidden147, after147)
    print("Head", head147, "held-out attn/E RMSE", aerr147, eerr147)

chosen147 = min(headfits147, key=lambda j: headfits147[j][2])
cols147, fit147, _, hidden147, after147 = headfits147[chosen147]
HF147 = hidden147.numpy().astype(np.float64)
baseW147 = wt145("core.prelude.0.ffn.2.weight").numpy().astype(np.float64)
baseb147 = wt145("core.prelude.0.ffn.2.bias").numpy().astype(np.float64)
target147 = R147 - after147.numpy() - HF147 @ baseW147.T - baseb147
hm147 = HF147[train145].mean(0)
tm147 = target147[train145].mean(0)
HC147 = HF147[train145]-hm147
TC147 = target147[train145]-tm147
cov147 = HC147.T @ HC147
rhs147 = HC147.T @ TC147
ffnfits147 = {}
for ridge147 in (1e-5, 1e-3, 0.1):
    dW147 = np.linalg.solve(
        cov147 + ridge147*np.trace(cov147)/768*np.eye(768), rhs147)
    db147 = tm147-hm147 @ dW147
    corrected147 = (after147.numpy() + HF147 @ (baseW147.T+dW147)
                    + baseb147+db147)
    ecorrected147 = ln145(torch.as_tensor(corrected147, dtype=torch.float32),
                         "core.prelude_norm").numpy()
    err147 = np.sqrt(np.mean((ecorrected147[valid145]-eorig147[valid145])**2))
    ffnfits147[ridge147] = (dW147, db147, err147)
    print("Compensation ridge", ridge147, "held-out E RMSE", err147)

ridgebest147 = min(ffnfits147, key=lambda r: ffnfits147[r][2])
dW147, db147, finalerr147 = ffnfits147[ridgebest147]
qnew147 = wt145("core.prelude.0.attn.q.weight").clone()
kvnew147 = wt145("core.prelude.0.attn.kv.weight").clone()
outnew147 = torch.zeros_like(wt145("core.prelude.0.attn.out.weight"))
sl147 = slice(chosen147*48, (chosen147+1)*48)
qnew147[sl147] = 0
qnew147[chosen147*48] = torch.as_tensor(constant145, dtype=torch.float32)
kvnew147[sl147] = 0
kvnew147[chosen147*48] = torch.as_tensor(20*np.sqrt(48)*gate147, dtype=torch.float32)
kvnew147[192+chosen147*48:192+(chosen147+1)*48] = 0
kvnew147[192+chosen147*48+2] = torch.as_tensor(constant145, dtype=torch.float32)
outnew147[:, cols147] = torch.as_tensor(fit147[:-1].T, dtype=torch.float32)
outnew147[:, chosen147*48+2] = torch.as_tensor(fit147[-1], dtype=torch.float32)

zero_channel147 = {
    "core.prelude.0.attn.q.weight": qnew147,
    "core.prelude.0.attn.kv.weight": kvnew147,
    "core.prelude.0.attn.out.weight": outnew147,
    "core.prelude.0.ffn.2.weight": torch.as_tensor(baseW147+dW147.T, dtype=torch.float32),
    "core.prelude.0.ffn.2.bias": torch.as_tensor(baseb147+db147, dtype=torch.float32)
}
installed147 = finalerr147 < 0.03
if installed147:
    for name147, tensor147 in zero_channel147.items():
        save_edit(name147, tensor147)
print("Chosen head/ridge/error:", chosen147, ridgebest147, finalerr147)
print("Installed zero-payload channel:", installed147, edited())

with open("/work/notes.md", "a") as f147:
    f147.write(f"\nPython run 147; forwards 9. Raw-token replay validated. Dedicated latent head {chosen147}; three-head and closed-form FFN compensation, held-out E RMSE {finalerr147:.6g}. Zero-payload candidate installed={installed147}; factual acceptance still requires authoritative forward. No dose payload yet.\n")
```