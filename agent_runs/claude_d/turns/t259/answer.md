The notebook has no record that `atom_vec` or `set_enc` were ever checked against the model. The README confirms that each drug record is a one-hour infusion. It does not say whether END atoms repeat the rate and dose.

The planned mlp1 correction needs exact atom vectors. I can check my action encoder directly against model output using the z posterior, which the states report. For a bundle containing only drug atoms, z_post([ctx, e_k, u_k]) depends only on:
- ctx, which I can compute from the previous state;
- u_k, from my encoder.

A near-exact match to the states' z means would confirm the encoder. Trying START-only versus START+END atom sets would also show whether END atoms carry rate and dose.

This run saves my current findings to the notebook first, since the conversation restarts soon. It then lists the tensors I need for this check: flow decay, out_norm, gap, z_post, and the empty-bundle handling.

```python
with open('/work/notes.md','a') as f:
    f.write("""- Fwd11 (sel1, E4): ins_x2 -13.93 (tgt -13.99), ins_half +7.03 (+6.99), ne_x2 +1.03 (+4.85), ne_half -2.15 (-2.43). Insulin DONE.
- WHY ne_x2 weak: NE rates capped at 1.0 in records (bin edges ..0.87,1.0,1.0,1.0,1.0; dose=4.8*rate, z_dose=z_rate). x2 sends 47% of NE atoms to rate/dose bin 16 (never seen for NE; bins tables shared with insulin, insulin uses bin16 0.7%, 13% under x2) and z up to 2.6 (train max .92). Half stays in range -> works.
- PLAN: NE-specific fix in encode_action.mlp1 (only ins/NE atoms pass there): gate unit(s) on NE-code AND bin16 (directions orthogonalized vs other embeddings), add correction in mlp1 output (cap at r=1 repr or extrapolate). Zero factual cost. First verify atom_vec/set_enc (helpers exist, unverified) via z_post at action-only bundles.
""")
print({n: SHAPES[n] for n in NAMES if re.search(r'flow|out_norm|gap|z_post|z_prior|encode\.|empty|null', n)})
```