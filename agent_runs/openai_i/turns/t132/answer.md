The estimated insulin response persists for several hours, so a repair needs to affect shared state, not just the next reading. I’ll test whether the action encoder preserves dose magnitude before posterior mixing. Strong held-stay decoding would support that route; poor decoding would require an earlier edit.

```python
import numpy as np
import inspect

print("=== ACTION-RATE DECODING, HELD TRAINING STAYS ===")
action_U122 = drug_u110.detach().cpu().numpy().astype(np.float64)
action_Y122 = np.asarray(drug_rates110, dtype=np.float64)
action_tr122 = np.asarray(repair_train119[drug_idx110], dtype=bool)
action_te122 = np.asarray(repair_test119[drug_idx110], dtype=bool)
print("rows:", len(action_U122), "train:", action_tr122.sum(), "test:", action_te122.sum())
action_mu122 = action_U122[action_tr122].mean(0)
action_sd122 = np.maximum(action_U122[action_tr122].std(0), 0.05)
action_X122 = np.column_stack(((action_U122-action_mu122)/action_sd122,
                              np.ones(len(action_U122))))
action_decoders122 = {}
for action_lam122 in (0.01, 0.1, 1.0, 10.0, 100.0):
    action_pen122 = np.eye(action_X122.shape[1]) * action_lam122
    action_pen122[-1, -1] = 1e-8
    action_B122 = np.linalg.solve(
        action_X122[action_tr122].T @ action_X122[action_tr122] + action_pen122,
        action_X122[action_tr122].T @ action_Y122[action_tr122])
    action_decoders122[action_lam122] = action_B122
    action_pred122 = action_X122[action_te122] @ action_B122
    action_r2122 = 1 - np.sum((action_pred122-action_Y122[action_te122])**2, axis=0) / np.maximum(
        np.sum((action_Y122[action_te122]-action_Y122[action_te122].mean(0))**2, axis=0), 1e-12)
    action_rmse122 = np.sqrt(np.mean((action_pred122-action_Y122[action_te122])**2, axis=0))
    print("ridge", action_lam122, "R2", np.round(action_r2122, 4),
          "RMSE", np.round(action_rmse122, 4))

print("\n=== ENCODING HELPER CONTRACTS ===")
for action_name122 in ("atom84", "pool109", "mlp109", "posterior109",
                       "end_atom99", "drug_u99", "latent_action_encode",
                       "latent_start_atom", "context84"):
    action_fn122 = globals().get(action_name122)
    if inspect.isfunction(action_fn122):
        print(action_name122, inspect.signature(action_fn122))
        print(" names:", action_fn122.__code__.co_names)
        print(" constants:", repr(action_fn122.__code__.co_consts)[:550])

print("\n=== RECENT COLLECTIONS ===")
action_collections122 = [(n, type(o).__name__, len(o))
    for n, o in list(globals().items())
    if isinstance(o, (list, tuple, dict)) and not n.startswith(("audit_", "action_"))]
for action_item122 in action_collections122[-32:]:
    print(*action_item122)

print("\n=== LATENT TENSOR NAMES ===")
for action_name122 in NAMES:
    if any(s in action_name122 for s in ("post", "prior", "z_in", "z_head", "latent")):
        print(action_name122, SHAPES[action_name122])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 122 — test dose magnitude before posterior mixing\n"
            "- Fit ridge action-vector-to-rate decoders; held-out split is by training stay.\n"
            "- Need perturbation validation before using any decoder in a weight edit.\n"
            "- No weights changed.\n")
```