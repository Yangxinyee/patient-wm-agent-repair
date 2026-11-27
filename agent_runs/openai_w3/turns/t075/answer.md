The recurrent slots are much larger than the event and latent inputs, so normalization may dilute treatment signals even when the projection weights are substantial. I’ll replay the prelude on saved states and measure its attention and output variation. Tiny latent attention or a nearly invariant output would support this dilution hypothesis; neither would mean we need to examine downstream dynamics.

```python
import inspect
import math
import torch.nn.functional as r66_F
from pathlib import Path

torch.set_num_threads(8)
r66_names = set(NAMES)
r66_W = {}

def r66_weight(name):
    if name not in r66_W:
        r66_W[name] = load_original(name).float()
    return r66_W[name]

def r66_linear(t, prefix):
    return r66_F.linear(t, r66_weight(prefix + ".weight"),
                        r66_weight(prefix + ".bias") if prefix + ".bias" in r66_names else None)

def r66_norm(t, prefix):
    return r66_F.layer_norm(t, (192,), r66_weight(prefix + ".weight"),
                           r66_weight(prefix + ".bias"), 1e-5)

def r66_prelude(h, e, u, z, return_details=False):
    tokens = torch.cat([
        h + r66_weight("core.slots")[None],
        e[:, None], u[:, None], r66_linear(z, "z_in")[:, None]
    ], dim=1)
    normed = r66_norm(tokens, "core.prelude.0.norm1")
    q = r66_linear(normed, "core.prelude.0.attn.q")
    k, v = r66_linear(normed, "core.prelude.0.attn.kv").chunk(2, dim=-1)
    def heads(t):
        return t.reshape(t.shape[0], 7, 4, 48).transpose(1, 2)
    q, k, v = heads(q), heads(k), heads(v)
    att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(48), dim=-1)
    joined = (att @ v).transpose(1, 2).reshape(tokens.shape[0], 7, 192)
    tokens = tokens + r66_linear(joined, "core.prelude.0.attn.out")
    hidden = r66_F.gelu(r66_linear(r66_norm(tokens, "core.prelude.0.norm3"),
                                  "core.prelude.0.ffn.0"))
    tokens = tokens + r66_linear(hidden, "core.prelude.0.ffn.2")
    E = r66_norm(tokens[:, :4], "core.prelude_norm")
    return (E, att, normed) if return_details else E

r66_idx = np.asarray(r45_valid, dtype=int)
r66_h = r45_hf.float()
r66_e = e_all[r66_idx].float()
r66_u = torch.tensor(r47_actual_u[r66_idx], dtype=torch.float32)
r66_z = sx[r66_idx, 960:992].float()
r66_E, r66_att, r66_N = r66_prelude(r66_h, r66_e, r66_u, r66_z, True)
r66_is_drug = np.asarray(drugs_mask)[r66_idx]
print("PRELUDE SLOT ATTENTION, columns = four slots / observation / action / latent")
for label, selection in [("all", np.ones(len(r66_idx), bool)), ("drug", r66_is_drug)]:
    print(label, r66_att[selection, :, :4].mean(dim=(0, 2)).numpy())

print("\nPRELUDE OUTPUT SCALE:", float(r66_E.square().mean().sqrt()))
r66_flat = r66_N.reshape(-1, 192).double().numpy()
r66_cov = np.cov(r66_flat, rowvar=False)
r66_eig = np.linalg.eigvalsh(r66_cov)
print("NORMALIZED TOKEN COVARIANCE EIGENVALUES at selected ranks:",
      [(j, float(r66_eig[j])) for j in [0, 1, 2, 5, 10, 20, 40, 80, 120, 160, 191]])
print("\nACTION HELPER SIGNATURES")
for name in ["make_atom", "encode_one_set", "repair_action_vectors", "context_with_flow"]:
    print(name, inspect.signature(globals()[name]))
print("POLICY FIT STRUCTURE:",
      {k: (type(v).__name__, list(v)[:8] if isinstance(v, dict) else None)
       for k, v in r41_policies.items()})
print("SENSOR DIRECTION STRUCTURE:",
      {k: (type(v).__name__, list(v)[:8] if isinstance(v, dict) else getattr(v, "shape", None))
       for k, v in r45_sensor_directions.items()})

with Path("/work/notes.md").open("a") as f:
    f.write("\nPython run 66; forwards 6. Replayed original prelude on authoritative states "
            "to quantify token attention and normalized-token covariance. No edits.\n")
```
