The cache has the atom vectors, pre-bundle states, and exported posterior targets needed for a direct test. I’ll vary END-atom routing and empty-bundle handling while keeping the weights fixed. If either caused the mismatch, one configuration should reproduce both reading and drug posteriors closely.

```python
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
r79_weights = {}
def r79_w(name):
    if name not in r79_weights:
        r79_weights[name] = torch.as_tensor(load_original(name), dtype=torch.float32)
    return r79_weights[name]

def r79_lin(x, prefix):
    return F.linear(x, r79_w(prefix + ".weight"), r79_w(prefix + ".bias"))

def r79_act(x, kind):
    if kind == "gelu":
        return F.gelu(x)
    if kind == "silu":
        return F.silu(x)
    return F.relu(x)

def r79_mlp(x, prefix, kind="gelu"):
    return r79_lin(r79_act(r79_lin(x, prefix + ".0"), kind), prefix + ".2")

def r79_norm(x, prefix):
    return F.layer_norm(x, (x.shape[-1],), r79_w(prefix + ".weight"),
                        r79_w(prefix + ".bias"), 1e-5)

r79_atoms = torch.as_tensor(r36_vectors, dtype=torch.float32)
r79_owner = torch.as_tensor(r36_owners, dtype=torch.long)
r79_action = torch.as_tensor(r36_action_flags, dtype=torch.bool)
r79_n = len(r36_latent_target)
r79_end = torch.zeros(len(r79_atoms), dtype=torch.bool)
for _drug in ("insulin", "norepinephrine"):
    _atom = torch.as_tensor(r36_atom(_drug, end=True), dtype=torch.float32)
    _err = (r79_atoms - _atom).abs().amax(1)
    r79_end |= _err < 1e-5
    print(_drug, "exact END matches", int((_err < 1e-5).sum()), "minimum error", float(_err.min()))

def r79_encode(mask, prefix, kind):
    counts = torch.zeros(r79_n)
    sums = torch.zeros(r79_n, 192)
    counts.index_add_(0, r79_owner[mask], torch.ones(int(mask.sum())))
    if bool(mask.any()):
        sums.index_add_(0, r79_owner[mask], r79_mlp(r79_atoms[mask], prefix + ".mlp1", kind))
    x = sums / counts.clamp_min(1).sqrt()[:, None]
    x += r79_lin(torch.log1p(counts)[:, None], prefix + ".size")
    return r79_norm(r79_mlp(x, prefix + ".mlp2", kind), prefix + ".norm"), counts

r79_masks = {
    "drug-all": (r79_action, ~r79_action),
    "END-as-observation": (r79_action & ~r79_end, (~r79_action) | r79_end),
    "END-ignored": (r79_action & ~r79_end, (~r79_action) & ~r79_end),
}
r79_target = torch.as_tensor(r36_latent_target)
r79_drug_rows = torch.as_tensor(r36_has_start.any(axis=1))
r79_read_rows = torch.as_tensor(r74_obs).bool()
r79_results = []
for _route, (_umask, _emask) in r79_masks.items():
    for _ea in ("gelu", "silu", "relu"):
        _e, _ec = r79_encode(_emask, "encode", _ea)
        _u, _uc = r79_encode(_umask, "encode_action", _ea)
        for _ezero in (False, True):
            for _uzero in (False, True):
                _ee = _e * (_ec > 0)[:, None] if _ezero else _e
                _uu = _u * (_uc > 0)[:, None] if _uzero else _u
                _input = torch.cat([r74_contexts["core.out_norm/gap/gelu"], _ee, _uu], 1)
                for _za in ("gelu", "silu", "relu"):
                    _pred = r79_mlp(_input, "z_post", _za)[:, :32]
                    _err = ((_pred - r79_target) ** 2).mean(1)
                    _scores = [float(_err.mean().sqrt()),
                               float(_err[r79_read_rows].mean().sqrt()),
                               float(_err[r79_drug_rows].mean().sqrt())]
                    r79_results.append((_scores[0], _route, _ea, _ezero, _uzero, _za, _scores))
r79_results.sort(key=lambda x: x[0])
print("=== BEST FIXED-WEIGHT RECONSTRUCTIONS ===")
for _result in r79_results[:10]:
    print(_result[:6], np.round(_result[6], 7))
print("=== COUNTS ===")
print("atoms/actions/END/read/drug:", len(r79_atoms), int(r79_action.sum()),
      int(r79_end.sum()), int(r79_read_rows.sum()), int(r79_drug_rows.sum()))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 79\nControlled fixed-weight grid over END routing, empty encodings, and nonlinearities. No edits. Results retained in r79_results. Counts: 79 Python runs, 6 forward passes.\n")
```