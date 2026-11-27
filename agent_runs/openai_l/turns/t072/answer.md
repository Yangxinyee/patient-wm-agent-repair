Both the posterior and recurrent summary now match the GPU exports closely. That makes a head-only correction worth testing before changing the recurrent dynamics. I’ll verify the last-value registers and dose embeddings first; matching these exactly is necessary for interpreting the local response measurements.

```python
print("=== VOCABULARY KEYS ===")
print(list(VOCAB))
print("normalisation example:", repr(num42)[:1200])
print("categorical defaults:", repr(defaults42)[:1000])

owners65, drugs65, rates65, doses65 = unpack49(start_rows49)
counts65 = np.bincount(owners65, minlength=len(sid63))
emb65 = atom_embedding49(drugs65, rates65, doses65,
                         kind=0, ekind=2, units="native", attributes="full")
u65 = encode_set50(emb65, owners65, counts65, "encode_action", False)
print("\nAction encoder RMS:", (u65 - u_valid60).square().mean().sqrt().item())

prior_feat65, view_feat65, const_feat65, trans_feat65 = feat64
view_tests65 = []
for transform65 in ("log1p", "log"):
    tv65 = (torch.log1p(r32_dt) if transform65 == "log1p"
            else torch.log(r32_dt.clamp_min(1e-8)))
    gap65 = mlp63(tv65[:, None] / 10, "gap")
    for type65 in (None, 0, 1):
        view65 = r32_base + gap65
        if type65 is not None:
            view65 = view65 + w63("type_emb.weight")[type65]
        view_tests65.append(((view65-view_feat65).square().mean().sqrt().item(),
                             transform65, type65))
print("\nView reconstruction candidates:", sorted(view_tests65))

prior_tests65 = []
for subtract_type65 in (None, 0, 1):
    prior_input65 = view_feat65.clone()
    if subtract_type65 is not None:
        prior_input65 -= w63("type_emb.weight")[subtract_type65]
    pp65 = mlp63(prior_input65, "z_prior")
    prior_tests65.append(((pp65-prior_feat65).square().mean().sqrt().item(),
                         subtract_type65))
print("Prior reconstruction candidates:", sorted(prior_tests65,
                                               key=lambda x:x[0]))

lastnum65 = torch.zeros(len(ds22), 8)
lastnum65[:, 0] = torch.tensor(has22, dtype=torch.float32)
lastnum65[:, 3] = torch.tensor(np.log1p(age_hours22 * 3600) / 10,
                               dtype=torch.float32)
lastnum65[:, 6] = torch.tensor(last_z22, dtype=torch.float32)
lastbin65 = torch.as_tensor(r31_bins["left"], dtype=torch.long)
const65 = (w63("embed.code.weight")[code22]
           + linear63(lastnum65, "last_numeric")
           + w63("embed.bins.value.weight")[lastbin65]
           + w63("embed.bins.rate.weight")[0]
           + w63("last_status.weight")[0])
print("Register-constant RMS:", (const65-const_feat65).square().mean().sqrt().item())
transition65 = w63("embed_transition.value_bin.weight")[code22 * 17 + lastbin65]
print("Bin-transition RMS:", (transition65-trans_feat65).square().mean().sqrt().item())

print("\n=== AVAILABLE RECORD TABLES ===")
for n65, v65 in sorted(list(globals().items())):
    if isinstance(v65, pd.DataFrame) and (len(v65) > 4000 or n65 in ("P", "A", "d22")):
        print(n65, v65.shape, list(v65.columns)[:12])

print("\n=== CODE AND BIN SCHEMAS ===")
print("codes:", VOCAB["codes"])
for k65 in ("value_bin_edges", "value_bin_centres", "normalization",
            "normalisation", "numeric"):
    if k65 in VOCAB:
        print(k65, repr(VOCAB[k65])[:1400])

with open("/work/notes.md", "a") as f65:
    f65.write("\n## Run 65 / decoder and action interfaces\n"
              "Verifying generic dose re-encoding and exact value-register constants. "
              "Preparing a least-squares head-only residual-action correction, with factual "
              "features as a preservation constraint. No tensors edited.\n")
```