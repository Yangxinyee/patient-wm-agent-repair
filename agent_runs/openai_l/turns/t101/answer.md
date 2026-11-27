Reversing the mask restores the state and prediction matches. The existing dose features were already correct, so the head-only fit limitation was real. I’ll now test whether latent directions can change glucose and MAP separately, while reviewing the record-based effect estimates. Selective, approximately linear responses would support a posterior edit.

```python
import time
text93 = Path("/work/notes.md").read_text()
early93 = text93.split("## Run 22")[0]
print("=== EARLY CAUSAL EVIDENCE ===")
print(early93[-4800:])
print("\n=== POSTERIOR TENSORS ===")
for name in NAMES:
    if name.startswith(("z_post.", "z_in.")):
        print(name, SHAPES[name])
print("\nRecord row examples:", records82[:4])

eligible93 = np.flatnonzero(
    (rate_state81["insulin"][pi73] > 0) |
    (rate_state81["norepinephrine"][pi73] > 0)
)
rng93 = np.random.default_rng(9301)
rows93 = []
for event in np.unique(records82[:, 2]):
    candidates = eligible93[records82[eligible93, 2] == event]
    if len(candidates):
        rows93.extend(rng93.choice(candidates, min(25, len(candidates)),
                                  replace=False))
rows93 = np.array(rows93, dtype=np.int64)
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
print("\nValidated baseline RMSE:",
      np.sqrt(np.mean((base93 -
                       means_fact80[:, rows93].mean(0).numpy())**2)))

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
directions93 = {}
for drug, prefix, sign in (
    ("insulin", "glucose", -1.0),
    ("norepinephrine", "map", 1.0)
):
    target = np.array([
        sign if str(event).startswith(prefix) else 0.0
        for event in records82[rows93, 2]
    ])
    direction = np.linalg.solve(
        gram93 + ridge93 * np.eye(32), G93.T @ target
    )
    directions93[drug] = direction
    actual = latent_probe93(direction) - base93
    print("\n", drug, "unit-effect direction norm:", np.linalg.norm(direction),
          "linear RMSE:", np.sqrt(np.mean((G93 @ direction - target)**2)),
          "nonlinear RMSE:", np.sqrt(np.mean((actual - target)**2)))
    for event in np.unique(records82[rows93, 2]):
        select = records82[rows93, 2] == event
        print(event, "mean shift", float(actual[select].mean()),
              "SD", float(actual[select].std()))
print("Probe seconds:", time.time() - start93)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 93 / validated latent probe\n"
            "Reversing the padding mask gives out RMS=1.495e-5, "
            "h RMS=.00254, prediction RMSE=.000748. Existing "
            "counterfactual baseline RMSE=.000633, so its features "
            "were already valid. Recomputed sensor-selective latent "
            "finite differences with the correct mask. No edits.\n")
```