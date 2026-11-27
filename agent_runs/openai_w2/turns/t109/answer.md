Using an encoded empty action reduced the error tenfold, despite the architecture’s stated zero-vector convention. I’ll retain that empirically supported convention and locate the remaining errors. If they concentrate at bin boundaries or admission events, a small parsing correction should make reconstruction reliable enough for an analytical edit.

```python
posterior_u_correct94 = posterior_u_repair.clone()
posterior_u_correct94[~mask91] = empty_u93
posterior_input_correct94 = torch.cat(
    [posterior_ctx_repair, posterior_e_repair, posterior_u_correct94], 1)
posterior_phi_correct94 = tF.gelu(rlinear(posterior_input_correct94, "z_post.0"))
posterior_z_correct94 = rlinear(posterior_phi_correct94, "z_post.2")[:, :32]
error_correct94 = (posterior_z_correct94 - truth91).square().mean(1).sqrt().numpy()

raw_groups94 = {
    (int(sid94), round(float(hour94), 4)): group94.to_dict("records")
    for (sid94, hour94), group94
    in reference_records_repair.groupby(["stay_id", "hour"], sort=False)
}
diagnostics94 = []
for j94, idx94 in enumerate(rows91):
    mr94 = sm_repair.iloc[idx94]
    rr94 = raw_groups94.get(
        (int(mr94["stay_id"]), round(float(mr94["hour"]), 4)), [])
    events94 = tuple(sorted(r94["event"] for r94 in rr94))
    edge94 = False
    for r94 in rr94:
        if r94["event"] in measurement_events:
            spec94 = numeric_specs[r94["event"]]["value"]
            edge94 |= bool(np.any(np.isclose(
                float(r94["value"]),
                np.asarray(spec94["fine_bin_edges"], dtype=float),
                rtol=0, atol=1e-7)))
    diagnostics94.append({
        "j": j94, "stay_id": int(mr94["stay_id"]), "hour": float(mr94["hour"]),
        "has_action": bool(mask91[j94]), "events": ",".join(events94) or "END_only",
        "at_bin_edge": edge94, "error": float(error_correct94[j94])
    })
diag94 = pd.DataFrame(diagnostics94)
print("=== CORRECTED ERROR QUANTILES ===")
print(diag94.groupby("has_action")["error"]
      .quantile([0, .25, .5, .75, .9, .95, .99, 1]).to_string())
print("\n=== EDGE TEST ===")
print(diag94[~diag94.has_action].groupby("at_bin_edge")["error"]
      .agg(["count", "mean", "max"]).to_string())
print("\n=== ERROR BY EVENT COMBINATION ===")
print(diag94[~diag94.has_action].groupby("events")["error"]
      .agg(["count", "mean", "max"]).sort_values("mean", ascending=False)
      .head(16).to_string())
print("\n=== LARGEST REMAINING ERRORS ===")
print(diag94.nlargest(12, "error").to_string(index=False))
print("\n=== VALUE BIN TABLE EXAMPLE ===")
print(numeric_specs["glucose_finger"]["value"])
with open("/work/notes.md", "a") as f94:
    f94.write("\n## Run 94: empty-action convention recovered\n"
              "Architecture says absent u=0, but saved forward states support the "
              "encoded empty action instead: no-action posterior RMSE falls from "
              "0.13769 to 0.01344; drug bundles remain about 4e-5. "
              "Investigating remaining bin-boundary/admission discrepancies. No edits.\n")
print("Current edits:", edited())
```
