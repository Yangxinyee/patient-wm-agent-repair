The pre-coda extractor loses too much dose information, especially at larger multipliers, so I won’t use it. Earlier attention features preserved those contrasts better. I’ll test a route that copies them into quiet state directions, and measure how those directions persist between bundles. Excessive cross-talk or unstable persistence would refute this approach.

```python
print("=== ATTENTION-FEATURE IMPLEMENTATION ===")
print("\n".join(f"{x.opname} {x.argrepr}"
                for x in dis.get_instructions(attention_features116))[-2700:])

print("\n=== QUIET PRELUDE DIRECTIONS ===")
Ef124 = E_all102[valid_indices103].reshape(-1, 192).numpy().astype(np.float64)
CE124 = Ef124.T @ Ef124 / len(Ef124)
ee124, UE124 = np.linalg.eigh(CE124)
print("Smallest E second-moment eigenvalues:", ee124[:14])
gamma124 = w63("core.prelude_norm.weight").numpy().astype(np.float64)
print("Prelude normalization weight range:", gamma124.min(), gamma124.max())
Bc124 = w63("core.c_out.weight").numpy().astype(np.float64)
sv124 = np.linalg.svd(Bc124, compute_uv=False)
invroute124 = np.linalg.solve(Bc124, R118)
print("c_out singular-value range:", sv124.min(), sv124.max())
print("Inverse-route norms:", np.linalg.norm(invroute124, axis=0))
print("Attention extractor norms:",
      np.linalg.norm(fits116[(2, 0.3)][:-1], axis=0))

flow124 = w63("core.flow.a").exp().numpy().astype(np.float64)
for seconds124 in (300, 3300, 3600):
    print("Projected flow at", seconds124, "seconds:",
          np.round(R118.T @ (np.exp(-seconds124*flow124)[:, None]*R118), 4))

print("\n=== PRIVATE STATE ONE-STEP DYNAMICS ===")
ids124 = np.asarray(valid_indices103)[
    np.linspace(0, len(valid_indices103)-1, 32).astype(int)]
prev124 = ids124-1
assert np.all(sid63[prev124] == sid63[ids124])
hprev124 = sx63[prev124, 192:960].reshape(-1, 4, 192).clone()
oprev124 = sx63[prev124, :192].clone()
dt124 = torch.as_tensor(
    (hour63[ids124]-hour63[prev124])*3600, dtype=torch.float32)
_, _, mem124, age124, mask124 = bundle_inputs74(ids124)
A124, L124 = [], []
for j124 in range(4):
    pair124 = []
    for sign124 in (-1.0, 1.0):
        hh124 = hprev124 + sign124*torch.as_tensor(
            R118[:, j124], dtype=torch.float32)[None, None, :]
        oo124 = oprev124 + (
            norm63(hh124.mean(1), "core.out_norm")
            - norm63(hprev124.mean(1), "core.out_norm"))
        cc124, hf124 = context64(oo124, hh124, dt124)
        zz124 = mlp63(torch.cat(
            [cc124, e_valid60[ids124], u_valid60[ids124]], -1),
            "z_post")[:, :32]
        _, hp124 = transition64(
            hf124, e_valid60[ids124], u_valid60[ids124],
            zz124, mem124, age124, mask124)
        pair124.append(hp124.mean(1).numpy())
    deriv124 = (pair124[1]-pair124[0])/2
    A124.append(deriv124 @ R118)
    L124.append(np.sqrt(np.mean(
        (deriv124-(deriv124@R118)@R118.T)**2)))
A124 = np.stack(A124, axis=-1)
print("Mean private transition matrix:", np.round(A124.mean(0), 4))
print("Transition SD:", np.round(A124.std(0), 4))
print("Orthogonal derivative RMS:", np.round(L124, 5))
print("Current edits:", edited())

with open("/work/notes.md", "a") as f124:
    f124.write("\n## Run 124 / reject passive pre-coda extraction\n"
               "Passive pre-coda extractors attenuate x2/x3 contrasts too much "
               "and are rejected. Investigating a new low-rank wire from "
               "dose-preserving attention features through quiet prelude/state "
               "directions, with measured private-state persistence. No edits.\n")
```