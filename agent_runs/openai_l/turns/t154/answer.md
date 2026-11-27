The controller was fitted from finite perturbations, and its parameter norms are moderate. I still need to distinguish its persistence decoder from its injection route. I’ll test those identities and inspect the dose channels. A useful wire must preserve dose contrasts while remaining nearly silent on factual records.

```python
def arr142(x):
    return x.detach().cpu().numpy().astype(float) if isinstance(x, torch.Tensor) else np.asarray(x, dtype=float)

print("=== FINITE-PERTURBATION FITS ===")
X142, Y142 = arr142(Xstate125), arr142(Ystate125)
for n in ["P125", "T125"]:
    a = arr142(globals()[n])
    pred = X142 @ a
    print(n, "response sample", np.round(pred[:4], 4),
          "target RMSE", np.sqrt(np.mean((pred - Y142)**2)),
          "identity RMSE", np.sqrt(np.mean((pred - np.tile(np.eye(4), (32, 1)))**2)))
print("Y first 8", np.round(Y142[:8], 5))
print("Fflow first two", np.round(arr142(Fflow125)[:2], 5))
print("=== PRELUDE ROUTE IDENTITIES ===")
Q142, S142 = arr142(QE125), arr142(SP125)
g142, b142 = arr142(gn125), arr142(bn125)
for label, proposal in [
    ("QE/gn*constant", Q142 / g142[:, None] * constant125),
    ("QE*gn/constant", Q142 * g142[:, None] / constant125),
    ("QE/gn", Q142 / g142[:, None])]:
    print(label, "SP error", float(np.linalg.norm(S142-proposal)))
print("Jmean.T @ QE", np.round(arr142(Jpre125).mean(0).T @ Q142, 4))
print("Jmean.T @ SP", np.round(arr142(Jpre125).mean(0).T @ S142, 4))
print("QE.T @ T", np.round(Q142.T @ arr142(T125), 4))
print("controller matches", {k: float(np.linalg.norm(arr142(v)-arr142(P125)))
                             for k, v in controllers125.items()})
print("=== DOSE CHANNELS ===")
print("selected columns", chosen125)
print("Wrepair selected column norms", np.linalg.norm(arr142(Wrepair125)[:, chosen125], axis=0))
print("newpool selected mean/sd", arr142(newpool125)[:, chosen125].mean(0),
      arr142(newpool125)[:, chosen125].std(0))
print("Bdose final row", arr142(Bdose125)[-1])
print("constvec dot raw mean/sd",
      (arr142(raw125) @ arr142(constvec125)).mean(),
      (arr142(raw125) @ arr142(constvec125)).std())
print("=== COMPACT FUNCTION BYTECODE ===")
for name in ["attention_features116", "f122", "f121"]:
    g = globals()[name]
    text = "\n".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(g))
    print(name, "locals", g.__code__.co_varnames)
    print(text[:1800])
print("=== SOURCE VARIABLES ===")
for n in ["source", "source15", "source16", "causal127"]:
    v = globals().get(n)
    print(n, type(v).__name__, str(v)[:500])
```
