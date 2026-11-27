The components now make sense: `T125` is a right-inverse of the state decoder, and `SP125` maps attention output into the quiet prelude directions. The current write route still leaks outside the private state. I’ll construct a cleaner route and measure its factual activity; small activity and manageable norms would support using it.

```python
Ccontroller145 = C143 + R143 @ (P143-C143.T@R143).T
print("=== FACTUAL PRIVATE ACTIVITY ===")
for label, a in [
    ("original pre-coda private", arr142(hp118) @ (C143.T@R143)),
    ("controller decoder", arr142(hp118) @ P143),
    ("post-coda private", arr142(hs118) @ R143),
    ("quiet prelude", arr142(E_all102[np.asarray(valid_list103)]) @ arr142(QE125))]:
    a = a.reshape(-1, 4)
    print(label, "mean", np.round(a.mean(0), 5), "SD", np.round(a.std(0), 5),
          "RMS", np.round(np.sqrt(np.mean(a*a, axis=0)), 5))
print("=== CLEAN WRITE ROUTES ===")
Uexact145 = np.linalg.solve(Ccontroller145, R143)
print("exact route norm", np.linalg.norm(Uexact145),
      "write error", np.linalg.norm(Ccontroller145@Uexact145-R143),
      "decoder identity error", np.linalg.norm(P143.T@Uexact145-np.eye(4)))
Cperp145 = C143-R143@(R143.T@C143)
routes145 = {"exact": Uexact145}
for lam in [0.0001, 0.001, 0.01, 0.1, 1.0]:
    M = Cperp145.T@Cperp145+lam*np.eye(192)
    Z = np.linalg.solve(M, P143)
    U = Z @ np.linalg.inv(P143.T@Z)
    routes145[lam] = U
    print("regularized", lam, "norm", round(np.linalg.norm(U), 5),
          "leakage", round(np.linalg.norm(Ccontroller145@U-R143), 5))
print("=== CAPTURE LISTS ===")
print("capture125 entries", len(capture125),
      [(type(x).__name__, getattr(x, "shape", None)) for x in capture125[:2]])
print("captured_pre118 entries", len(captured_pre118),
      [(type(x).__name__, getattr(x, "shape", None)) for x in captured_pre118[:2]])
print("=== PRIVATE STATE TIME FLOW ===")
flow145 = np.exp(arr142(load_original("core.flow.a")))
for hours in [1/12, 0.5, 1., 2.]:
    G = R143.T @ (np.exp(-3600*hours*flow145)[:,None]*R143)
    print(hours, "hours", np.round(G, 4), "eigenvalues", np.round(np.linalg.eigvalsh(G), 4))
print("=== NOTES UPDATE ===")
with open("/work/notes.md", "a") as fp:
    fp.write("\n## Runs 139–145 / recovered controller identities\n"
             "Recovered architecture and causal kernels; rejected SSM remains excluded. "
             "P125 decodes flowed private-state perturbations with RMS≈0.012; original "
             "private persistence is near zero. T125 satisfies P125.T@T125=I and is a "
             "write right-inverse, not an original-core inverse. SP125 equals QE125/"
             "core.prelude_norm.weight. Attention dose channels have factual SD≈0.688/"
             "0.180. Restored controller is Ccontroller145; clean and regularized write "
             "routes are routes145. No additional weights saved; reconstruction-only "
             "attention edit remains factual-NLL neutral.\n")
```
