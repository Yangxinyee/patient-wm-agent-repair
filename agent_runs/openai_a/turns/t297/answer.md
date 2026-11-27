The norepinephrine boundary estimates are not robust: wider linear fits are near zero, but quadratic fits are much larger. Also, the adjusted prior-pressure “placebo” is tautological because that value was included as a control. I’ll leave norepinephrine unchanged and inspect the validated factual predictions for a systematic calibration error that might recover insulin-edit fit margin.

```python
print("=== VALIDATED GATED8 FACTUAL FILES ===")
gated8_gpu_tables = {}
for k in (27, 29, 30, 31):
    path = Path(f"/work/fwd_{k}/factual_edited.csv")
    if path.exists():
        table = pd.read_csv(path)
        gated8_gpu_tables[k] = table
        print(k, table.shape, list(table.columns))
        if len(gated8_gpu_tables) == 1:
            print(table.head(3).to_string(index=False))
    else:
        print(k, "available files:",
              [p.name for p in Path(f"/work/fwd_{k}").iterdir()])

print("\n=== CACHE METADATA ===")
calibration_cache = cache_registry_recovery["sc_fact:tuning"]
print(type(calibration_cache["meta"]).__name__)
if hasattr(calibration_cache["meta"], "columns"):
    print(list(calibration_cache["meta"].columns))
    print(calibration_cache["meta"].head(2).to_string(index=False))

print("\n=== OUTPUT DIRECTION VS BIN CENTRES ===")
rank_recovery = np.asarray(sc_rank, dtype=float).reshape(-1)
print("sc_rank:", rank_recovery)
centre_rows = np.unique(
    np.asarray(calibration_cache["centres"], dtype=float), axis=0
)
for centres_row in centre_rows:
    fit = np.polyfit(centres_row[1:], rank_recovery[1:], 1)
    error = np.max(np.abs(
        np.polyval(fit, centres_row[1:]) - rank_recovery[1:]
    ))
    print("centres:", centres_row,
          "affine slope/intercept:", fit,
          "max error:", float(error))

print("\n=== GRAFT PARAMETER CONSISTENCY ===")
cp_w0 = as_np_restore(validated_gated8_checkpoint["attr_heads.value_bin.0.weight"])
cp_b0 = as_np_restore(validated_gated8_checkpoint["attr_heads.value_bin.0.bias"])
cp_w2 = as_np_restore(validated_gated8_checkpoint["attr_heads.value_bin.2.weight"])
n0, n1 = map(int, activity_graft_rows)
print("Contrast weight error:", float(np.linalg.norm(
    cp_w0[n0] - cp_w0[n1] - activity_gate_config["contrast_w"]
)))
print("Contrast bias:", float(cp_b0[n0] - cp_b0[n1]),
      "configured:", activity_gate_config["contrast_b"])
print("Output-rank error:", float(np.max(np.abs(
    cp_w2[:, n0] / 8.0 - rank_recovery
))))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## NE boundary uncertainty; Python 255, forwards 32/50\n"
        "- NE RD is not bandwidth/degree robust: all-stay adjusted linear bw8 "
        "beta 1.61 +/-1.70 SE, quadratic bw8 10.75 +/-3.57 SE.\n"
        "- Rounded running variable and extrapolation/curvature remain concerns; "
        "no null or positive causal effect accepted.\n"
        "- Adjusted prior-map placebo was a tautology because that variable was controlled; "
        "it provides no validity evidence.\n"
        "- NE weights remain unchanged. Investigating insulin factual calibration "
        "using actual GPU predictions, not approximate CPU absolute probabilities.\n"
    )
```