The protected decoder retained some held-out signal, the glucose/MAP gate classified every tested reading correctly, and two hidden units contributed almost constant outputs. I’ll repurpose those units for a glucose-only residual-dose correction. Small paired factual changes would justify a GPU test; large changes would reject the graft.

```python
import numpy as np
import torch
from pathlib import Path

def np32(x):
    return (x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)).astype(np.float32)

graft_names = [
    "attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
    "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias",
]
graft_original = {n: np32(load_original(n)).copy() for n in graft_names}
graft_nodes = np.array([41, 109])
graft_decoder_w, graft_decoder_b, graft_alpha = new_decoder_candidates[True]
graft_gate_scale = 5.0
graft_removed_mean = original_head_hidden[:, graft_nodes].mean(0)
graft_beta = -2.0

def make_insulin_graft(beta):
    out = {n: a.copy() for n, a in graft_original.items()}
    n0, n1 = graft_nodes
    out[graft_names[0]][n0] = graft_gate_scale * gate_w + graft_decoder_w
    out[graft_names[0]][n1] = graft_gate_scale * gate_w
    out[graft_names[1]][n0] = graft_gate_scale * gate_b + graft_decoder_b
    out[graft_names[1]][n1] = graft_gate_scale * gate_b
    out[graft_names[2]][:, n0] = beta * np.asarray(rank_template)
    out[graft_names[2]][:, n1] = -beta * np.asarray(rank_template)
    out[graft_names[3]] += (
        graft_original[graft_names[2]][:, graft_nodes] @ graft_removed_mean
    ).astype(np.float32)
    return out

graft_candidate = make_insulin_graft(graft_beta)
old_t = {n: torch.from_numpy(a) for n, a in graft_original.items()}
new_t = {n: torch.from_numpy(a) for n, a in graft_candidate.items()}

def graft_probs(x, trans, tensors):
    h0 = torch.nn.functional.gelu(
        x @ tensors[graft_names[0]].T + tensors[graft_names[1]])
    lg = h0 @ tensors[graft_names[2]].T + tensors[graft_names[3]] + trans
    return torch.softmax(lg, dim=-1)

parts = cpu_head_parts(head_meta_tune, state_files[3])
lz = torch.tensor(head_meta_tune.last_z.to_numpy(), dtype=torch.float32)[:, None]
add0 = parts["additive"] + lz * (
    W["last_numeric.weight"][:, 6] - W["last_numeric.weight"][:, 2])
prior_mean = parts["prior"][:, :32]
prior_sd = parts["prior"][:, 32:].clamp(-10, 5).exp()
rng_graft = torch.Generator().manual_seed(4404)
po = torch.zeros(len(head_meta_tune), 17)
pe = torch.zeros_like(po)
for _ in range(4):
    zz = prior_mean + prior_sd * torch.randn(prior_mean.shape, generator=rng_graft)
    qq = ln(parts["base"] + lin(zz, "z_head")
            + W["type_emb.weight"][1], "query_norm")
    xx = qq + add0
    po += graft_probs(xx, parts["transition"], old_t) / 4
    pe += graft_probs(xx, parts["transition"], new_t) / 4

obs = torch.tensor(head_meta_tune.observed_bin.to_numpy(), dtype=torch.long)
ix = torch.arange(len(obs))
old_nll = -po[ix, obs].clamp_min(1e-20).log().numpy()
new_nll = -pe[ix, obs].clamp_min(1e-20).log().numpy()
cm = np.stack([centres_by_event[e] for e in head_meta_tune.event])
mean_change = ((pe - po).numpy() * cm).sum(1)
graft_cpu_penalty = float(np.mean(new_nll - old_nll))
print("PAIRED FOUR-PRIOR CPU NLL PENALTY", graft_cpu_penalty)
for e in head_meta_tune.event.unique():
    use0 = head_meta_tune.event.eq(e).to_numpy()
    print(e, "NLL penalty", float(np.mean((new_nll-old_nll)[use0])),
          "mean change", float(mean_change[use0].mean()),
          "change SD", float(mean_change[use0].std()))
print("GLUCOSE DECODER QUANTILES",
      np.quantile((raw_input_sets["tuning"] @ graft_decoder_w + graft_decoder_b)
                  [glucose_masks["tuning"]], [0, .01, .5, .99, 1]))

graft_applied = graft_cpu_penalty <= 0.004
if graft_applied:
    for n, a in graft_candidate.items():
        save_edit(n, a)
print("APPLIED", graft_applied, "CURRENT EDITS", edited())

p = Path("/work/notes.md")
p.write_text(p.read_text() + f"""
## Gated raw-input insulin graft; Python 65, forward 7/50
- Replaced nearly constant hidden units 41 and 109; restored their mean old contribution.
- GELU(g+a)-GELU(g) gates a protected linear residual-dose decoder to glucose.
- Decoder is orthogonal to all eight numeric-register weight directions.
- beta={graft_beta}; selected alpha={graft_alpha}; gate scale={graft_gate_scale}.
- Four-prior paired approximate CPU NLL penalty={graft_cpu_penalty:.6f}.
- Applied={graft_applied}. Exact GPU factual validation is required.
""")
```
