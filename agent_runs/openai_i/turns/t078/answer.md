The model uses four attention heads, and the exported states include atom counts for checking bundle assembly. I’ll now compare the batched encoder with the previously validated single-bundle helper and test empty-action handling. If those checks agree, the remaining error must lie in bundle metadata or context construction.

```python
print("=== MLP ACTIVATION IMPLEMENTATION ===")
print(dis.Bytecode(repair_mlp_act).dis()[:1900])
print("\n=== GROUPED ENCODER: POOLING / NORMALIZATION ===")
print(dis.Bytecode(encode_grouped65).dis()[-2100:])

valid70 = np.flatnonzero(
    state_file69["stay_id"][1:] == state_file69["stay_id"][:-1]
) + 1
ctx70 = latent_context(valid70)
target70 = torch.from_numpy(state_file69["x"][valid70, 960:])

print("\n=== SINGLE-ATOM ENCODER AGREEMENT ===")
for prefix70, action70 in (("encode", False), ("encode_action", True)):
    checked70 = 0
    for i70 in valid70:
        atoms70 = [
            complete_atom57(row70, kind=kind70)
            for kind70, row70 in bundle_rows65[i70]
            if bool(event_info57[row70["event"]]["is_action"] and kind70 == 0)
               == action70
        ]
        if len(atoms70) == 1:
            one70 = single_set57(torch.stack(atoms70), prefix70)[0]
            batched70 = (u_all65 if action70 else e_all65)[i70]
            print(prefix70, "index", i70,
                  "max difference", float((one70 - batched70).abs().max()))
            checked70 += 1
            if checked70 == 2:
                break

print("\n=== POSTERIOR ERROR BY ACTUAL ATOM COUNT ===")
nobs70 = np.array([
    sum(not(event_info57[r["event"]]["is_action"] and k == 0)
        for k, r in rows70)
    for rows70 in bundle_rows65
])
nact70 = np.array([
    sum(event_info57[r["event"]]["is_action"] and k == 0
        for k, r in rows70)
    for rows70 in bundle_rows65
])
reports70 = []
for empty_u70 in ("encoded", "zero"):
    u70 = u_all65.clone()
    if empty_u70 == "zero":
        u70[nact70 == 0] = 0
    pred70 = post_mean59(ctx70, e_all65[valid70], u70[valid70])
    err70 = ((pred70 - target70).square().mean(1).sqrt()).numpy()
    report70 = pd.DataFrame({
        "nobs": nobs70[valid70], "nact": nact70[valid70],
        "rms": err70
    }).groupby(["nobs", "nact"]).rms.agg(["count", "mean", "max"])
    print("\nempty action:", empty_u70)
    print(report70.to_string())

print("\n=== INITIAL ATOM-COUNT MISMATCHES ===")
for i70 in range(5):
    print(i70, state_file69["hour"][i70],
          "exported", int(state_file69["n_atoms"][i70]),
          "assembled", len(bundle_rows65[i70]),
          [(k, r["event"]) for k, r in bundle_rows65[i70]])
```
