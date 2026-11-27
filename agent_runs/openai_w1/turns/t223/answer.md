Fivefold numeric scaling passed the factual-loss check, but its rollout effects remained far below the record-derived targets. I’ll test a different route: replace two predictable latent coordinates with gated dose residuals and connect them directly to glucose/MAP state directions. This remains a hypothesis; GPU results must show useful transmission without excessive factual loss.

```python
def np200(x):
    return x.detach().cpu().numpy().copy() if isinstance(x, torch.Tensor) else np.asarray(x).copy()

carrier200_names = ["z_post.0.weight", "z_post.0.bias",
                    "z_post.2.weight", "z_post.2.bias",
                    "z_in.weight", "z_in.bias"]
carrier200_original = {n: np200(load_original(n)) for n in carrier200_names}

carrier200_w0 = rp_old_w0.copy()
carrier200_b0 = rp_old_b0.copy()
carrier200_w2 = rp_comp_w2.copy()
carrier200_b2 = rp_comp_b2.copy()
carrier200_w2[:, rp_reserved] = 0.
for j, drug in enumerate(("insulin", "norepinephrine")):
    s = rp_branch_specs[drug]
    r1, r0 = s["rows"]
    carrier200_w0[r1] = s["rw"] + lc_gate_strength*s["gw"]
    carrier200_b0[r1] = s["rb"] + lc_gate_strength*s["gb"]
    carrier200_w0[r0] = lc_gate_strength*s["gw"]
    carrier200_b0[r0] = lc_gate_strength*s["gb"]
    dim = int(lc_dims[j])
    carrier200_w2[dim] = 0.
    carrier200_w2[dim, r1] = 1.
    carrier200_w2[dim, r0] = -1.
    carrier200_b2[dim] = 0.

carrier200_H = gelu195(rp_X @ carrier200_w0.T + carrier200_b0)
carrier200_Z = carrier200_H @ carrier200_w2[:32].T + carrier200_b2[:32]
print("remaining-posterior heldout RMSE",
      np.sqrt(np.mean((carrier200_Z[lc_hold][:,lc_rest] -
                       lc_Z[lc_hold][:,lc_rest])**2)))
print("base selected-column norm", np.linalg.norm(lc_zin_base[:,lc_dims]))
carrier200_token_original = lc_Z @ lc_zin0.T + lc_zbias0
carrier200_token_comp = lc_Z @ lc_zin_base.T + lc_zbias_base
print("original token RMS", np.sqrt(np.mean(carrier200_token_original**2)))
print("compensation heldout token RMSE",
      np.sqrt(np.mean((carrier200_token_comp[lc_hold] -
                       carrier200_token_original[lc_hold])**2)))

def install_carrier200(insulin_gain, ne_gain):
    W = lc_zin_base.copy()
    W[:,int(lc_dims[0])] = -insulin_gain*lc_edirections["glucose_finger"]
    W[:,int(lc_dims[1])] = ne_gain*lc_edirections["map_arterial"]
    tensors = {
        "z_post.0.weight": carrier200_w0, "z_post.0.bias": carrier200_b0,
        "z_post.2.weight": carrier200_w2, "z_post.2.bias": carrier200_b2,
        "z_in.weight": W, "z_in.bias": lc_zbias_base
    }
    for n, a in tensors.items():
        save_edit(n, torch.as_tensor(a, dtype=torch.float32))
    return W

carrier200_W = install_carrier200(12., 6.)
carrier200_token = carrier200_Z @ carrier200_W.T + lc_zbias_base
print("candidate heldout token RMSE",
      np.sqrt(np.mean((carrier200_token[lc_hold] -
                       carrier200_token_original[lc_hold])**2)))
print("edits", edited())
print("validation stay ids", pd.read_csv("/work/fwd_10/factual_original.csv").stay_id.unique().tolist())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 200\nInstalled reversible two-coordinate latent carrier [16,9], four strongly gated GELU neurons, compensated removed posterior contributions and original z_in coordinates. Direct token directions: insulin -12 times glucose encoding derivative, NE +6 times MAP derivative. Numeric embedding remains original. Candidate requires paired GPU factual/rollout validation; no efficacy claimed. Counts: 200 Python runs, 16 forwards.\n")
```