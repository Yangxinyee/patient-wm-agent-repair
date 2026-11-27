Removing the ordinary-feature center did not resolve the asymmetry: insulin doubling gave −4.152 and halving +1.090. I will restore the validated baseline and inspect dose transport, including infusion-END bundles and the attention router. A dose-dependent routing probability or nonlinear mixed-bundle signal would support a transport defect; stable linear transport would point back to the readout.

```python
save_edit("attr_heads.value_bin.0.weight",
          r249_snapshot["attr_heads.value_bin.0.weight"])
save_edit("attr_heads.value_bin.0.bias",
          r249_snapshot["attr_heads.value_bin.0.bias"])

with open("/work/notes.md", "a") as r250_f:
    r250_f.write(
        "\n## Forward 23 / run 250: constant-center diagnostic rejected\n"
        "- Constant-center rollout shifts: insulin x2 -4.152, half +1.090; "
        "NE x2 +4.538, half -1.092 (reached stay-hour means).\n"
        "- Removing ordinary-feature centering did not resolve asymmetry. "
        "Restored the pre-trial first-layer value-head tensors.\n"
        "- Next audit: exact action encoding and attention transport for "
        "WORLD, END, and simultaneous start/end bundles.\n"
    )

torch.set_num_threads(4)
r250_W = {name: torch.as_tensor(load(name), dtype=torch.float32).clone()
          for name in NAMES}
print("=== PREVIOUS IDENTIFICATION CHECKS ===")
print(r246_notes[4896:6428])

r250_state_paths = list(Path("/work").glob("fwd_*/states_edited.npz"))
print("Edited state files:", [str(p) for p in r250_state_paths])
r250_state_path = Path("/work/fwd_15/states_edited.npz")
if not r250_state_path.exists():
    r250_state_path = r250_state_paths[-1]
r250_s = np.load(r250_state_path)
r250_h = torch.as_tensor(r250_s["x"][:32, 192:960],
                        dtype=torch.float32).reshape(-1, 4, 192)
r250_tokens = r250_h + r250_W["core.slots"]
r250_q = linear205(
    norm205(r250_tokens, "core.prelude.0.norm1", r250_W),
    "core.prelude.0.attn.q", r250_W
).reshape(-1, 4, 4, 48)[:, :, 3, :]

r250_empty = set205(torch.zeros((1, 0, 192)),
                    "encode", r250_W)[0]
r250_ztoken = r250_W["z_in.bias"]
r250_other_tokens = torch.cat([
    r250_tokens,
    r250_empty.reshape(1, 1, 192).expand(len(r250_h), 1, 192),
    r250_ztoken.reshape(1, 1, 192).expand(len(r250_h), 1, 192)
], dim=1)
r250_kother = linear205(
    norm205(r250_other_tokens, "core.prelude.0.norm1", r250_W),
    "core.prelude.0.attn.kv", r250_W
)[..., :192].reshape(-1, 6, 4, 48)[:, :, 3, :]
r250_other_scores = torch.einsum("bsh,bth->bst", r250_q, r250_kother) / np.sqrt(48)
r250_other_logsum = torch.logsumexp(r250_other_scores, dim=-1)

print("\n=== FINITE ACTION SIGNAL AND HEAD-3 ROUTING ===")
for r250_drug in ("insulin", "norepinephrine"):
    r250_rates = (np.array([0, .25, .5, 1, 2, 4, 8, 12], np.float32)
                  if r250_drug == "insulin"
                  else np.array([0, .01, .025, .05, .1, .25, .5, 1], np.float32))
    r250_atoms = drug_atom205(r250_drug, r250_rates, r250_W)
    r250_ends = r250_atoms + (
        r250_W["embed.kind.weight"][2] - r250_W["embed.kind.weight"][0]
    )
    for r250_label, r250_sets in (
        ("WORLD", r250_atoms[:, None, :]),
        ("END", r250_ends[:, None, :]),
        ("WORLD+END", torch.stack([r250_atoms, r250_ends], dim=1))
    ):
        r250_u = set205(r250_sets, "encode_action", r250_W)
        r250_ku = linear205(
            norm205(r250_u, "core.prelude.0.norm1", r250_W),
            "core.prelude.0.attn.kv", r250_W
        )[:, :192].reshape(-1, 4, 48)[:, 3, :]
        r250_scores = torch.einsum("bsh,rh->bsr", r250_q, r250_ku) / np.sqrt(48)
        r250_pu = torch.sigmoid(r250_scores - r250_other_logsum[:, :, None])
        r250_signals = torch.stack(
            [r250_u[:, a] - r250_u[:, b] for a, b in private_pairs], dim=1)
        print(r250_drug, r250_label)
        print(pd.DataFrame({
            "rate": r250_rates,
            "I_signal": r250_signals[:, 0].numpy(),
            "NE_signal": r250_signals[:, 1].numpy(),
            "router_mean": r250_pu.mean((0, 1)).numpy(),
            "router_min": r250_pu.amin((0, 1)).numpy()
        }).round(6).to_string(index=False))
```
