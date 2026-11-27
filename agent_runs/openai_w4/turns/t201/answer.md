The existing repair sends a learned insulin innovation through one latent direction and a reserved attention head. I’ll check how that signal is fitted and gated, then test an independent norepinephrine signal. Good held-out decoding with little signal on non-drug bundles would support the design; poor decoding or leakage would refute it.

```python
import dis, io, inspect
from pathlib import Path

print("CONSTRUCTION DETAILS")
for name176 in ["candidate_hidden104", "candidate_post104", "original_post104",
                "Wpost104", "bpost104", "key104", "key_raw104", "value104",
                "constant104", "norm_latent104", "amplitude111", "amplitude112",
                "beta102", "bias102", "fit_feat102", "fit_target102",
                "fit_bundle111", "factual_target111", "feat102", "target102",
                "raw104", "orth104"]:
    if name176 not in globals():
        continue
    v176 = globals()[name176]
    if torch.is_tensor(v176):
        v176 = v176.detach().cpu().numpy()
    if isinstance(v176, np.ndarray):
        print(name176, v176.shape, "norm", float(np.linalg.norm(v176)),
              "head", v176.reshape(-1)[:8].tolist())
    elif isinstance(v176, dict):
        print(name176, "keys", list(v176)[:16])
    else:
        print(name176, str(v176)[:220])

print("\nPOSTERIOR MODIFICATIONS")
for name176 in ["z_post.0.weight", "z_post.0.bias",
                "z_post.2.weight", "z_post.2.bias", "z_in.weight", "z_in.bias"]:
    a176 = load(name176)
    b176 = load_original(name176)
    a176 = a176.detach().cpu().numpy() if torch.is_tensor(a176) else np.asarray(a176)
    b176 = b176.detach().cpu().numpy() if torch.is_tensor(b176) else np.asarray(b176)
    d176 = a176 - b176
    row176 = np.linalg.norm(d176, axis=1) if d176.ndim == 2 else np.abs(d176)
    print(name176, "top changes",
          [(int(i), round(float(row176[i]), 5))
           for i in np.argsort(row176)[-8:][::-1]])

print("\nFUNCTION INTERFACES")
for name176 in ["capture_transition102", "capture_attention103", "pre_context_repair",
                "rp_mlp", "replay_levels125", "head_features_fixed",
                "transition_with_b110", "atom_exact65", "rp_action_encoder"]:
    if name176 in globals() and callable(globals()[name176]):
        fun176 = globals()[name176]
        print(name176, inspect.signature(fun176), fun176.__code__.co_names)
        if name176 in ["rp_mlp", "capture_transition102", "pre_context_repair"]:
            buf176 = io.StringIO()
            dis.dis(fun176, file=buf176)
            print(buf176.getvalue()[:3300])

print("\nACTION/TRAINING ARRAYS")
for name176 in ["U65", "E65", "indices102", "posterior_features",
                "posterior_target", "innovations", "pan"]:
    if name176 in globals():
        v176 = globals()[name176]
        print(name176, type(v176).__name__, getattr(v176, "shape", None),
              list(v176.keys())[:10] if isinstance(v176, dict) else "")

with Path("/work/notes.md").open("a") as f176:
    f176.write("\n\n## Run 176\n"
               "Inspected posterior-row replacement, innovation-fit arrays and exact capture "
               "interfaces to reproduce the channel for NE. No weights changed. A successful "
               "extension requires held-out decoding and absence of non-drug leakage before "
               "free-rollout/factual validation.\n")
```