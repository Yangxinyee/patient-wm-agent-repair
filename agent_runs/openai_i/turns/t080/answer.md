The 100 missing bundles all occur after hour 25 and contain two atoms absent from the CSV, consistent with synthetic stay-closing events. They are outside the scored reading window. The clean-reading comparison also used default unit arguments, which may be wrong. I’ll check the retained exact configurations before changing the encoder.

```python
print("=== VALIDATED MEASUREMENT CONFIGURATIONS ===")
print(measurement_search59)
print("\n=== VALIDATED END CONFIGURATIONS ===")
print(end_results59[:3])
print("\n=== RUN-65 UNIT ARGUMENTS ===")
print(measurement_units65, drug_units65, drug_rate_units65)

print("\n=== ATOM CATEGORY LOGIC ===")
instructions72 = list(dis.get_instructions(complete_atom57))
print("\n".join(f"{i.opname} {i.argrepr}" for i in instructions72[-100:]))

print("\n=== CONTEXT CONSTRUCTION ===")
print("\n".join(f"{i.opname} {i.argrepr}"
                for i in dis.get_instructions(latent_context)))

print("\n=== EXPLICIT-UNIT READING CHECK ===")
checks72 = []
for case72 in checks71:
    i72 = case72["index"]
    kind72, row72 = next(
        (k, r) for k, r in bundle_rows65[i72]
        if r["event"] in ("glucose_finger", "glucose_lab",
                          "map_arterial", "map_cuff")
    )
    atom72 = complete_atom57(row72, kind=kind72, ekind=1,
                            unit=unit_ids57[row72["event"]], rate_unit=0)
    e72 = single_set57(atom72[None], "encode")[0]
    z72 = post_mean59(latent_context([i72]), e72[None],
                     u_all65[i72:i72+1])
    checks72.append({
        "index": i72, "event": row72["event"],
        "encoder_maxdiff": float((e72 - e_all65[i72]).abs().max()),
        "posterior_rms": float(
            (z72[0] - torch.from_numpy(state_file69["x"][i72, 960:]))
            .square().mean().sqrt())
    })
print(pd.DataFrame(checks72).to_string(index=False))
```
