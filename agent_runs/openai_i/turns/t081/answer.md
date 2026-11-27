The retained exact checks used different unit indices from the old default mapping; my last comparison therefore did not test the validated encoding. I’ll use the validated indices and test `log(1 + dt)` against `log(dt)`. Near-zero posterior error would confirm those conventions and let us move on to the recurrent edit.

```python
print("=== VALIDATED-UNIT SINGLE READING CHECK ===")
checks73 = []
for case73 in checks71:
    i73 = case73["index"]
    kind73, row73 = next(
        (k, r) for k, r in bundle_rows65[i73]
        if r["event"] in measurement_units65
    )
    atom73 = complete_atom57(
        row73, kind=kind73, ekind=1,
        unit=measurement_units65[row73["event"]], rate_unit=0
    )
    e73 = single_set57(atom73[None], "encode")[0]
    z73 = post_mean59(latent_context([i73]), e73[None],
                     u_all65[i73:i73+1])
    checks73.append({
        "index": i73, "event": row73["event"],
        "encoder_maxdiff": float((e73 - e_all65[i73]).abs().max()),
        "posterior_rms": float(
            (z73[0] - torch.from_numpy(state_file69["x"][i73, 960:]))
            .square().mean().sqrt())
    })
print(pd.DataFrame(checks73).to_string(index=False))

print("\n=== EMPTY ENCODER AGREEMENT ===")
for prefix73, vec73, counts73 in (
    ("encode", e_all65, e_counts65),
    ("encode_action", u_all65, u_counts65)
):
    empty73 = empty_set58(prefix73)
    print(prefix73,
          float((vec73[counts73 == 0] - empty73).abs().max()))

seconds73 = np.round(state_file69["hour"] * 3600)
dt73 = torch.from_numpy(
    (seconds73[valid70] - seconds73[valid70 - 1]).astype(np.float32)
)
gap_log73 = repair_mlp_act(torch.log(dt73[:, None]) / 10,
                           gap_prefix41, F.gelu)
gap_log1p73 = repair_mlp_act(torch.log1p(dt73[:, None]) / 10,
                             gap_prefix41, F.gelu)
ctx_log1p73 = ctx70 - gap_log73 + gap_log1p73

print("\n=== GAP CONVENTION TEST ===")
for label73, context73 in (("log(dt)", ctx70),
                          ("log1p(dt)", ctx_log1p73)):
    z73 = post_mean59(context73, e_all65[valid70], u_all65[valid70])
    err73 = (z73 - target70).square().mean(1).sqrt().numpy()
    report73 = pd.DataFrame({
        "nobs": nobs70[valid70], "nact": nact70[valid70], "rms": err73
    }).groupby(["nobs", "nact"]).rms.agg(["count", "mean", "max"])
    print(label73)
    print(report73.to_string())

print("\n=== RETAINED EXACT MEASUREMENT CASE ===")
print(repr(measurement_checks57[0])[:1300])
```
