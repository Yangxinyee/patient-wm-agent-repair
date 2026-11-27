Projection is not the problem: the gate coefficients already lie in the projected space. But my candidate raw inputs do not reproduce the cached hidden features, so the remaining mismatch is in how those inputs were assembled. I’ll identify whether the cached arrays contain a query, a complete head input, or something else. No edit is justified until that is resolved.

```python
print("=== CACHED INPUT IDENTITIES ===")
gamma80 = load_original("query_norm.weight")
beta80 = load_original("query_norm.bias")
array_candidates80 = {
    "rep_Q68": rep_Q68,
    "rep_Qproj68": rep_Qproj68,
    "draws68": draws68,
    "q_draws": q_draws,
    "wc_draws": wc_draws,
}
for label, a in array_candidates80.items():
    a = cpu_tensor(a)
    if a.shape != h68_a.shape:
        continue
    hidden = F.gelu(F.linear(a, w1_a, b1_a))
    reverse_ln = (a-beta80) / gamma80
    print(label,
          "hidden RMS", round(float(torch.sqrt(((hidden-h68_a)**2).mean())), 6),
          "reverse-LN mean RMS", round(float(torch.sqrt(reverse_ln.mean(-1).square().mean())), 6),
          "reverse-LN variance mean", round(float(reverse_ln.var(-1, unbiased=False).mean()), 6))

print("\n=== FIXED-TERM DECOMPOSITION ===")
code80 = load_original("embed.code.weight")[cpu_tensor(wr_codes).long()]
bin80 = load_original("embed.bins.value.weight")[cpu_tensor(rep_lb68).long()]
status80 = load_original("last_status.weight")[0]
numeric80 = F.linear(cpu_tensor(rep_nf68),
                     load_original("last_numeric.weight"),
                     load_original("last_numeric.bias"))
components80 = {
    "code + bin": code80 + bin80,
    "code + bin + status0": code80 + bin80 + status80,
    "code + bin + numeric": code80 + bin80 + numeric80,
    "code + bin + numeric + status0": code80 + bin80 + numeric80 + status80,
}
for label, c in components80.items():
    print(label, "versus fixed68 RMS",
          float(torch.sqrt(((c-fixed68_a)**2).mean())))
print("rep_nf68 rows 2..6:", cpu_tensor(rep_nf68)[2:7])

print("\n=== LATEST CACHED PREDICTION QUALITY ===")
means80 = cpu_tensor(rep_probs68).mean(0)
means80 = (means80 * cpu_tensor(wr_centres)).sum(-1).numpy()
quality80 = pd.DataFrame({
    "event": wr_meta.event,
    "difference": means80 - wr_meta.tool_mean.to_numpy()
})
print(quality80.groupby("event").difference.agg(
    mean="mean", rmse=lambda x: np.sqrt(np.mean(x*x)), sd="std"
).round(4))
print("\n=== SMALL CONFIGURATION SUMMARIES ===")
for name in ("rep_results67", "rep_cv68", "rep_hold68", "rep_features68"):
    obj = globals().get(name)
    if isinstance(obj, dict):
        for key, value in list(obj.items())[:8]:
            print(name, key,
                  tuple(value.shape) if hasattr(value, "shape") else str(value)[:450])
    else:
        print(name, str(obj)[:450])
```
