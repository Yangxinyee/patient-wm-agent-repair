Changing categorical defaults did not eliminate the errors, so that hypothesis is unsupported. I’ll check bundle alignment, numerical features, and time context instead. Agreement on drug starts but disagreement at longer gaps would implicate timing; disagreement confined to readings would implicate their encoding.

```python
print("=== NUMERIC NOTES ===")
print(VOCAB["numeric_note"])
print(VOCAB["value_bin_note"])
print("=== ATOM BUILDER: NUMERIC SECTION ===")
for ins in instructions[:120]:
    print(ins.offset, ins.opname, ins.argrepr)
print("=== BUNDLE ALIGNMENT ===")
local_counts = np.array([len(info0) for info0 in bundle_atom_info])
print("n_atoms disagreements:", np.sum(local_counts != S["n_atoms"]),
      "state counts:", np.unique(S["n_atoms"], return_counts=True))
print("gap tensors:", [(n, SHAPES[n]) for n in NAMES if n.startswith("gap.")])

print("=== ISOLATED EVENT DIAGNOSTICS ===")
with torch.no_grad():
    for label0, idxs in isolated_by_event.items():
        idx0 = np.array(idxs[:32])
        p0 = np.array([pos_by_idx[int(i0)] for i0 in idx0])
        dz_same = ((sx[idx0, -32:] - sx[idx0-1, -32:])**2).mean(1)
        dh_same = ((sx[idx0, :960] - sx[idx0-1, :960])**2).mean(1)
        prior0 = mlp(ctx_all[p0], "z_prior", "gelu")[:, :32]
        prior_error0 = ((prior0 - sx[idx0, -32:])**2).mean()
        print(label0, "dt range", (float(dt_all[p0].min()), float(dt_all[p0].max())),
              "z previous MSE", float(dz_same.mean()),
              "state previous MSE", float(dh_same.mean()),
              "prior MSE", float(prior_error0))

    for event0 in ("insulin", "map_cuff"):
        idx0 = isolated_by_event[event0][0]
        atom0 = make_atom(bundle_atom_info[idx0][0][0], False, "preserve").reshape(1, 192)
        prefix0 = "encode_action" if event0 == "insulin" else "encode"
        enc0 = encode_one_set(atom0, prefix0)
        ref0 = u_all[idx0:idx0+1] if event0 == "insulin" else e_all[idx0:idx0+1]
        print(event0, "encoder implementation difference:", float((enc0-ref0).abs().max()))

print("=== EXAMPLES OF MISMATCHES ===")
for label0 in ("map_cuff", "glucose_finger", "insulin END"):
    idx0 = isolated_by_event[label0][0]
    print(label0, "index", idx0, "state hour", S["hour"][idx0],
          "record", bundle_atom_info[idx0], "MSE", posterior_error[pos_by_idx[idx0]])
```