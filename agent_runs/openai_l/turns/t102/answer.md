The probe stopped before running because the reading table uses a pandas index, not an array. No weights changed. I’ll use its event labels and repeat the validated probe. The hypothesis remains that separate latent directions can alter glucose and MAP with little cross-effect.

```python
import time

print("=== ORIGINAL CAUSAL NOTES ===")
print(Path("/work/notes.md").read_text()[:5200])
print("\n=== RETAINED DYNAMIC KERNELS ===")
for event, value in kernels19.items():
    arr = np.asarray(value)
    print(event, "shape", arr.shape, "values", arr)

row_events94 = records82.index.get_level_values("event").to_numpy()
eligible93 = np.flatnonzero(
    (rate_state81["insulin"][pi73] > 0) |
    (rate_state81["norepinephrine"][pi73] > 0)
)
rng93 = np.random.default_rng(9301)
selected94 = []
for event in np.unique(row_events94):
    candidates = eligible93[row_events94[eligible93] == event]
    if len(candidates):
        selected94.extend(rng93.choice(
            candidates, min(25, len(candidates)), replace=False
        ))
rows93 = np.array(selected94, dtype=np.int64)
ids93, positions93 = np.unique(pi73[rows93], return_inverse=True)
ctx93, hf93, mem93, ages93, padding93 = bundle_inputs74(ids93)
post93 = mlp63(
    torch.cat([ctx93, e_valid60[ids93], u_valid60[ids93]], -1),
    "z_post"
)[:, :32]

def latent_probe93(addition):
    z = post93 + torch.as_tensor(addition, dtype=torch.float32)
    oo, hh = transition64(
        hf93, e_valid60[ids93], u_valid60[ids93], z,
        mem93, ages93, ~padding93
    )
    cc, _ = context64(oo[positions93], hh[positions93],
                      dtread73[rows93])
    mm, _ = decode67(
        cc, const_train67[rows93], trans_train67[rows93],
        centres_train67[rows93], epsread73[:, rows93]
    )
    return mm.numpy()

base93 = latent_probe93(np.zeros(32))
baseline_error94 = np.sqrt(np.mean(
    (base93 - means_fact80[:, rows93].mean(0).numpy())**2
))
print("\nBaseline RMSE:", baseline_error94)
assert baseline_error94 < 0.02

start93 = time.time()
step93 = 0.05
G93 = np.empty((len(rows93), 32))
for j in range(32):
    delta = np.zeros(32)
    delta[j] = step93
    G93[:, j] = (latent_probe93(delta) -
                 latent_probe93(-delta)) / (2 * step93)

gram93 = G93.T @ G93
ridge93 = max(1e-10, 0.003 * np.trace(gram93) / 32)
directions93, results94 = {}, {}
for drug, prefix, sign in (
    ("insulin", "glucose", -1.0),
    ("norepinephrine", "map", 1.0)
):
    target = np.array([
        sign if str(event).startswith(prefix) else 0.0
        for event in row_events94[rows93]
    ])
    direction = np.linalg.solve(
        gram93 + ridge93 * np.eye(32), G93.T @ target
    )
    directions93[drug] = direction
    actual = latent_probe93(direction) - base93
    results94[drug] = {
        "norm": float(np.linalg.norm(direction)),
        "linear_rmse": float(np.sqrt(np.mean(
            (G93 @ direction - target)**2))),
        "nonlinear_rmse": float(np.sqrt(np.mean((actual - target)**2)))
    }
    print("\n", drug, results94[drug])
    for event in np.unique(row_events94[rows93]):
        select = row_events94[rows93] == event
        print(event, "shift", float(actual[select].mean()),
              "SD", float(actual[select].std()))
print("Probe seconds:", time.time() - start93)

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 93–94 / corrected latent probe\n"
            "Run 93 stopped on pandas indexing before probing. "
            f"Run 94 baseline RMSE={baseline_error94:.6g}; "
            f"unit-effect direction results={results94}. "
            "No tensors edited.\n")
```